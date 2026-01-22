import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call(self, scope, equiv_set, expr, lhs):
    from numba.stencils.stencil import StencilFunc
    callee = expr.func
    callee_def = get_definition(self.func_ir, callee)
    if isinstance(callee_def, (ir.Global, ir.FreeVar)) and is_namedtuple_class(callee_def.value):
        return ArrayAnalysis.AnalyzeResult(shape=tuple(expr.args))
    if isinstance(callee_def, (ir.Global, ir.FreeVar)) and isinstance(callee_def.value, StencilFunc):
        args = expr.args
        return self._analyze_stencil(scope, equiv_set, callee_def.value, expr.loc, args, dict(expr.kws))
    fname, mod_name = find_callname(self.func_ir, expr, typemap=self.typemap)
    added_mod_name = False
    if isinstance(mod_name, ir.Var) and isinstance(self.typemap[mod_name.name], types.ArrayCompatible):
        args = [mod_name] + expr.args
        mod_name = 'numpy'
        added_mod_name = True
    else:
        args = expr.args
    fname = '_analyze_op_call_{}_{}'.format(mod_name, fname).replace('.', '_')
    if fname in UFUNC_MAP_OP:
        return self._analyze_broadcast(scope, equiv_set, expr.loc, args, None)
    else:
        try:
            fn = getattr(self, fname)
        except AttributeError:
            return None
        result = guard(fn, scope=scope, equiv_set=equiv_set, loc=expr.loc, args=args, kws=dict(expr.kws))
        if added_mod_name:
            expr.args = args[1:]
        return result