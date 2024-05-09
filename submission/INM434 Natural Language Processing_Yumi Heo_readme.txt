-------------------------------------------------------------------------------------
INM434 Natural Language Processing | Yumi Heo | Student ID: 230003122 
-------------------------------------------------------------------------------------
***As the moodle cannot upload the best trained models, ***
***Please use the shared google drive folder link below where you can find the test code, test sets, and saved best models to load and run them reliably***
***Only codes in ipynb files and test sets are uploaded on Moodle for submission***

*Google Colab Folder Link: https://drive.google.com/drive/folders/1Yn99YR6d5iJ79NYjdZwLLUTEPmucnNqH?usp=sharing
*Model Training Code Google Colab Link:https://colab.research.google.com/drive/1nTohqQt6vPr6GIj6mX9jmtQiaoiIGg07?usp=sharing
*Model Test Code Google Colab Link: https://colab.research.google.com/drive/1OgBrPqldusG9K2o68J3SBCF-pTexFih0?usp=sharing

=== Description ===

[Setup instructions for test]

1. Open the Google Colab Folder Link. The name of the folder is 1. NLP CW
2. If you would like to test the code, open the file name with (Model Test)INM434 Natural Language Processing_Yumi Heo_code.
3. In the test code, all trained best models and corresponding test sets will call directly using the code in the notebook. 
4. The folder LSTM stores all trained LSTM models including the best LSTM model, and stroes the test set.
5. The folder DistilBERT stores all trained DistilBERT models including the best DistilBERT model, and stroes the test sets.
6. Although the model and test set will load without any change in the test code, make sure to use the test set name with testset not testset2 to test the best DistilBERT model.

=== Environment & Package Spec ===

*Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
*Pandas version: 2.0.3
*PyTorch version: 2.2.1+cu121
*TensorFlow version: 2.15.0
*Keras version: 2.15.0
*Numpy version: 1.25.2
*nltk version: 3.8.1.
*Scikit-learn version: 1.2.2.
*Hugging Face Datasets is tested on Python 3.6+.
*Hugging Face Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+,

=== Uploaded Files in Google Drive Folder (1. NLP CW) ===

-- Code
*(Model_Training)INM434_Natural_Language_Processing_Yumi_Heo_code.ipynb
-> This is the code file for initial data analysis, training, tuning and selection of models.
*(Model_Test)INM434_Natural_Language_Processing_Yumi_Heo_code.ipynb
-> This is the code file for testing models.

-- Dataset
*** In forlder DistilBERT>test set
*testset.csv
-> This is the test set for the best DistilBERT model.

*** In forlder LSTM>test set
*X_test_padded.npy
-> This is the test set for the best LSTM model.
*y_test_array.npy
-> This is the test set for the best LSTM model.

-- Model
*** In forlder DistilBERT>
*processed_distilbert.bin
*distilBERT_model.pth
-> These are the best LSTM model with configs and weights.

*** In forlder LSTM>
*dropout_tuned_LSTM_word2vec_model.keras
->This is the best LSTM model.

=== Uploaded Files on Moodle ===
***In the code,testset zip file,

-- Code
*(Model_Training)INM434_Natural_Language_Processing_Yumi_Heo_code.ipynb
-> This is the code file for initial data analysis, training, tuning and selection of models.
*(Model_Test)INM434_Natural_Language_Processing_Yumi_Heo_code.ipynb
-> This is the code file for testing models.

-- Dataset
*testset.csv
-> This is the test set for the best DistilBERT model.
*X_test_padded.npy
-> This is the test set for the best LSTM model.
*y_test_array.npy
-> This is the test set for the best LSTM model.





