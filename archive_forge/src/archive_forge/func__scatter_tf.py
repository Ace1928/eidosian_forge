from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_tf(indices, array, new_dims):
    import tensorflow as tf
    indices = np.expand_dims(indices, 1)
    return tf.scatter_nd(indices, array, new_dims)