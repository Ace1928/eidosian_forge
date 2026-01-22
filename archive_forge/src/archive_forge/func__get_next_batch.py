import functools
import math
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _get_next_batch(generator):
    """Retrieves the next batch of input data."""
    try:
        generator_output = next(generator)
    except (StopIteration, errors.OutOfRangeError):
        return None
    if not isinstance(generator_output, tuple):
        generator_output = (generator_output,)
    if len(generator_output) not in [1, 2, 3]:
        raise ValueError('Output of generator should be a tuple of 1 or 2 or 3 elements: (input,) or (input, target) or (input, target, sample_weights). Received {}'.format(generator_output))
    return generator_output