import contextlib
import copy
import gc
import math
import os
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import base_tuner
from keras_tuner.src.engine import tuner_utils
def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
    if backend.config.backend() != 'tensorflow':
        return
    from tensorboard.plugins.hparams import api as hparams_api
    for callback in callbacks:
        if callback.__class__.__name__ == 'TensorBoard':
            logdir = self._get_tensorboard_dir(callback.log_dir, trial.trial_id, execution)
            callback.log_dir = logdir
            hparams = tuner_utils.convert_hyperparams_to_hparams(trial.hyperparameters, hparams_api)
            callbacks.append(hparams_api.KerasCallback(writer=logdir, hparams=hparams, trial_id=trial.trial_id))