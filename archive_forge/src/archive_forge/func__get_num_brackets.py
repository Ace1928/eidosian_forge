import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _get_num_brackets(self):
    epochs = self.max_epochs
    brackets = 0
    while epochs >= self.min_epochs:
        epochs = epochs / self.factor
        brackets += 1
    return brackets