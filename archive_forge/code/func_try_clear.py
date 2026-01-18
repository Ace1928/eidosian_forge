import json
from keras_tuner.src import backend
from keras_tuner.src.backend import keras
def try_clear():
    if IS_NOTEBOOK:
        display.clear_output()
    else:
        print()