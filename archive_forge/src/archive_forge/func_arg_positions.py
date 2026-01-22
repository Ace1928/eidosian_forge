import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
@property
@cached_per_instance
def arg_positions(self):
    """Returns a dict mapping arg names to their index positions."""
    call_fn_arg_positions = dict()
    for pos, arg in enumerate(self._arg_names):
        call_fn_arg_positions[arg] = pos
    return call_fn_arg_positions