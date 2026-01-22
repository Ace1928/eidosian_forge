from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _unwrap_arraybox(arraybox, max_depth=None, _n=0):
    if max_depth is not None and _n == max_depth:
        return arraybox
    val = getattr(arraybox, '_value', arraybox)
    if hasattr(val, '_value'):
        return _unwrap_arraybox(val, max_depth=max_depth, _n=_n + 1)
    return val