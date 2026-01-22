from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_element_add_tf(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    import tensorflow as tf
    if not isinstance(index[0], int):
        index = tuple(zip(*index))
    indices = tf.expand_dims(index, 0)
    value = tf.cast(tf.expand_dims(value, 0), tensor.dtype)
    return tf.tensor_scatter_nd_add(tensor, indices, value)