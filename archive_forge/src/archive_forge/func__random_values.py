import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def _random_values(self):
    """Fills the hyperparameter space with random values.

        Returns:
            A dictionary mapping hyperparameter names to suggested values.
        """
    collisions = 0
    while 1:
        hps = hp_module.HyperParameters()
        for hp in self.hyperparameters.space:
            hps.merge([hp])
            if hps.is_active(hp):
                hps.values[hp.name] = hp.random_sample(self._seed_state)
                self._seed_state += 1
        if self._duplicate(hps.values):
            collisions += 1
            if collisions > self._max_collisions:
                return None
            continue
        break
    return hps.values