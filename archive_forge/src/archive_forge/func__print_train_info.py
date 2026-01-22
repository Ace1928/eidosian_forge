import functools
import numpy as np
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _print_train_info(num_samples_or_steps, val_samples_or_steps, is_dataset):
    increment = 'steps' if is_dataset else 'samples'
    msg = 'Train on {0} {increment}'.format(num_samples_or_steps, increment=increment)
    if val_samples_or_steps:
        msg += ', validate on {0} {increment}'.format(val_samples_or_steps, increment=increment)
    print(msg)