from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _kron_tf(a, b):
    import tensorflow as tf
    a_shape = a.shape
    b_shape = b.shape
    if len(a_shape) == 1:
        a = a[:, tf.newaxis]
        b = b[tf.newaxis, :]
        return tf.reshape(a * b, (a_shape[0] * b_shape[0],))
    a = a[:, tf.newaxis, :, tf.newaxis]
    b = b[tf.newaxis, :, tf.newaxis, :]
    return tf.reshape(a * b, (a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]))