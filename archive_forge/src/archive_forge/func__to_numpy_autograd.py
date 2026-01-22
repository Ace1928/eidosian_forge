from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _to_numpy_autograd(x, max_depth=None, _n=0):
    if hasattr(x, '_value'):
        return _unwrap_arraybox(x, max_depth=max_depth, _n=_n)
    return x.numpy()