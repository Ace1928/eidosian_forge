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
def _find_unsafe_empty_inferred(func_ir, expr):
    unsafe_empty_inferred
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = get_definition(func_ir, callee)
    require(isinstance(callee_def, ir.Global))
    _make_debug_print('_find_unsafe_empty_inferred')(callee_def.value)
    return callee_def.value == unsafe_empty_inferred