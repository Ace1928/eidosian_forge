import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_build_tuple(self, scope, equiv_set, expr, lhs):
    for x in expr.items:
        if isinstance(x, ir.Var) and isinstance(self.typemap[x.name], types.ArrayCompatible) and (self.typemap[x.name].ndim > 1):
            return None
    consts = []
    for var in expr.items:
        x = guard(find_const, self.func_ir, var)
        if x is not None:
            consts.append(x)
        else:
            break
    else:
        out = tuple([ir.Const(x, expr.loc) for x in consts])
        return ArrayAnalysis.AnalyzeResult(shape=out, rhs=ir.Const(tuple(consts), expr.loc))
    return ArrayAnalysis.AnalyzeResult(shape=tuple(expr.items))