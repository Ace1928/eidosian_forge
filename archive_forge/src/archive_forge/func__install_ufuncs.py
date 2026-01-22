import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _install_ufuncs():
    for func_name in _ufuncs:
        _install_ufunc(func_name)