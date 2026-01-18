import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def weight_memory_size(weights):
    """Calculate the memory footprint for weights based on their dtypes.

    Args:
        weights: An iterable contains the weights to compute weight size.

    Returns:
        The total memory size (in Bytes) of the weights.
    """
    unique_weights = {id(w): w for w in weights}.values()
    total_memory_size = 0
    for w in unique_weights:
        if not hasattr(w, 'shape'):
            continue
        elif None in w.shape.as_list():
            continue
        weight_shape = np.prod(w.shape.as_list())
        per_param_size = w.dtype.size
        total_memory_size += weight_shape * per_param_size
    return total_memory_size