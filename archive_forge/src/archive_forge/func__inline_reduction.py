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
def _inline_reduction(self, work_list, block, i, expr, call_name):
    require(not self.parallel_options.reduction)
    require(call_name == ('reduce', 'builtins') or call_name == ('reduce', '_functools'))
    if len(expr.args) not in (2, 3):
        raise TypeError('invalid reduce call, two arguments are required (optional initial value can also be specified)')
    check_reduce_func(self.func_ir, expr.args[0])

    def reduce_func(f, A, v=None):
        it = iter(A)
        if v is not None:
            s = v
        else:
            s = next(it)
        for a in it:
            s = f(s, a)
        return s
    inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, reduce_func, work_list=work_list, callee_validator=callee_ir_validator)
    return True