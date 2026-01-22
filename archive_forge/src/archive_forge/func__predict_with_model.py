import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
def _predict_with_model(self, distribution, model, predict_dataset):
    return model.predict(predict_dataset, steps=PREDICT_STEPS)