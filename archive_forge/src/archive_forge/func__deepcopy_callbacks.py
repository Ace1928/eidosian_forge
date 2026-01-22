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
def _deepcopy_callbacks(self, callbacks):
    try:
        callbacks = copy.deepcopy(callbacks)
    except:
        raise errors.FatalValueError('All callbacks used during a search should be deep-copyable (since they are reused across trials). It is not possible to do `copy.deepcopy(%s)`' % (callbacks,))
    return callbacks