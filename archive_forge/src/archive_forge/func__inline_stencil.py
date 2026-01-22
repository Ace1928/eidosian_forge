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
def _inline_stencil(self, instr, call_name, func_def):
    from numba.stencils.stencil import StencilFunc
    lhs = instr.target
    expr = instr.value
    if isinstance(func_def, ir.Global) and func_def.name == 'stencil' and isinstance(func_def.value, StencilFunc):
        if expr.kws:
            expr.kws += func_def.value.kws
        else:
            expr.kws = func_def.value.kws
        return True
    require(call_name == ('stencil', 'numba.stencils.stencil') or call_name == ('stencil', 'numba'))
    require(expr not in self._processed_stencils)
    self._processed_stencils.append(expr)
    if not len(expr.args) == 1:
        raise ValueError('As a minimum Stencil requires a kernel as an argument')
    stencil_def = guard(get_definition, self.func_ir, expr.args[0])
    require(isinstance(stencil_def, ir.Expr) and stencil_def.op == 'make_function')
    kernel_ir = get_ir_of_code(self.func_ir.func_id.func.__globals__, stencil_def.code)
    options = dict(expr.kws)
    if 'neighborhood' in options:
        fixed = guard(self._fix_stencil_neighborhood, options)
        if not fixed:
            raise ValueError('stencil neighborhood option should be a tuple with constant structure such as ((-w, w),)')
    if 'index_offsets' in options:
        fixed = guard(self._fix_stencil_index_offsets, options)
        if not fixed:
            raise ValueError('stencil index_offsets option should be a tuple with constant structure such as (offset, )')
    sf = StencilFunc(kernel_ir, 'constant', options)
    sf.kws = expr.kws
    sf_global = ir.Global('stencil', sf, expr.loc)
    self.func_ir._definitions[lhs.name] = [sf_global]
    instr.value = sf_global
    return True