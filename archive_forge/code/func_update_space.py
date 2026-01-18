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
def update_space(self, hyperparameters):
    """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            hyperparameters: An updated `HyperParameters` object.
        """
    hps = hyperparameters.space
    new_hps = [hp for hp in hps if not self.hyperparameters._exists(hp.name, hp.conditions)]
    if new_hps and (not self.allow_new_entries):
        raise RuntimeError(f'`allow_new_entries` is `False`, but found new entries {new_hps}')
    if not self.tune_new_entries:
        return
    self.hyperparameters.merge(new_hps)