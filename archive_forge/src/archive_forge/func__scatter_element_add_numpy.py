from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_element_add_numpy(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    new_tensor = tensor.copy()
    new_tensor[tuple(index)] += value
    return new_tensor