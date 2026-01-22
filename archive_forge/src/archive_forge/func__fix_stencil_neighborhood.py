import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def _fix_stencil_neighborhood(self, options):
    """
        Extract the two-level tuple representing the stencil neighborhood
        from the program IR to provide a tuple to StencilFunc.
        """
    dims_build_tuple = get_definition(self.func_ir, options['neighborhood'])
    require(hasattr(dims_build_tuple, 'items'))
    res = []
    for window_var in dims_build_tuple.items:
        win_build_tuple = get_definition(self.func_ir, window_var)
        require(hasattr(win_build_tuple, 'items'))
        res.append(tuple(win_build_tuple.items))
    options['neighborhood'] = tuple(res)
    return True