import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
class StepTimingCallback(keras.callbacks.Callback):
    """A callback that times non-warmup steps of a Keras predict call."""

    def __init__(self):
        self.t0 = None
        self.steps = 0

    def on_predict_batch_begin(self, batch_index, _):
        if batch_index == 2:
            self.t0 = time.time()
        elif batch_index > 2:
            self.steps += 1

    def on_predict_end(self, _):
        self.tn = time.time()
        self.t_avg = (self.tn - self.t0) / self.steps