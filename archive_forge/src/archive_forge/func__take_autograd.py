from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _take_autograd(tensor, indices, axis=None):
    indices = _i('qml').numpy.asarray(indices)
    if axis is None:
        return tensor.flatten()[indices]
    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[tuple(fancy_indices)]