import collections
import copy
import csv
import json
import os
import re
import sys
import time
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_file_utils
from keras.src.distribute import worker_training_state
from keras.src.optimizers import optimizer
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import tf_utils
from keras.src.utils import version_utils
from keras.src.utils.data_utils import Sequence
from keras.src.utils.generic_utils import Progbar
from keras.src.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@keras_export('keras.callbacks.experimental.BackupAndRestore', v1=[])
@deprecation.deprecated_endpoints('keras.callbacks.experimental.BackupAndRestore')
class BackupAndRestoreExperimental(BackupAndRestore):
    """Deprecated. Please use `tf.keras.callbacks.BackupAndRestore` instead.

    Caution: `tf.keras.callbacks.experimental.BackupAndRestore` endpoint is
      deprecated and will be removed in a future release. Please use
      `tf.keras.callbacks.BackupAndRestore`.
    """

    def __init__(self, *args, **kwargs):
        logging.warning('`tf.keras.callbacks.experimental.BackupAndRestore` endpoint is deprecated and will be removed in a future release. Please use `tf.keras.callbacks.BackupAndRestore`.')
        super().__init__(*args, **kwargs)