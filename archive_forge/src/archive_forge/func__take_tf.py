from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _take_tf(tensor, indices, axis=None, **kwargs):
    tf = _i('tf')
    return tf.experimental.numpy.take(tensor, indices, axis=axis, **kwargs)