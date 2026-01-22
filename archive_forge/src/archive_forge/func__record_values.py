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
def _record_values(self, trial):
    hyperparameters = trial.hyperparameters
    hyperparameters.ensure_active_values()
    new_hash_value = self._compute_values_hash(hyperparameters.values)
    self._tried_so_far.add(new_hash_value)
    old_hash_value = self._id_to_hash[trial.trial_id]
    if old_hash_value != new_hash_value:
        self._id_to_hash[trial.trial_id] = new_hash_value
        if old_hash_value in self._tried_so_far:
            self._tried_so_far.remove(old_hash_value)