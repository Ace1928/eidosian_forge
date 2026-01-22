import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_static_getitem(self, scope, equiv_set, expr, lhs):
    var = expr.value
    typ = self.typemap[var.name]
    if not isinstance(typ, types.BaseTuple):
        result = self._index_to_shape(scope, equiv_set, expr.value, expr.index_var)
        if result[0] is not None:
            expr.index_var = result[0]
        return result[1]
    shape = equiv_set._get_shape(var)
    if isinstance(expr.index, int):
        require(expr.index < len(shape))
        return ArrayAnalysis.AnalyzeResult(shape=shape[expr.index])
    elif isinstance(expr.index, slice):
        return ArrayAnalysis.AnalyzeResult(shape=shape[expr.index])
    require(False)