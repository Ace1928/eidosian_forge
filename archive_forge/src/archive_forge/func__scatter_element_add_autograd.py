from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_element_add_autograd(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Since Autograd doesn't support indexing
    assignment, we have to be clever and use ravel_multi_index."""
    pnp = _i('qml').numpy
    size = tensor.size
    flat_index = pnp.ravel_multi_index(index, tensor.shape)
    if pnp.isscalar(flat_index):
        flat_index = [flat_index]
    if pnp.isscalar(value) or len(pnp.shape(value)) == 0:
        value = [value]
    t = [0] * size
    for _id, val in zip(flat_index, value):
        t[_id] = val
    return tensor + pnp.array(t).reshape(tensor.shape)