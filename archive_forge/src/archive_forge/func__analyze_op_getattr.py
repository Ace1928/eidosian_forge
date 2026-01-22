import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_getattr(self, scope, equiv_set, expr, lhs):
    if expr.attr == 'T' and self._isarray(expr.value.name):
        return self._analyze_op_call_numpy_transpose(scope, equiv_set, expr.loc, [expr.value], {})
    elif expr.attr == 'shape':
        shape = equiv_set.get_shape(expr.value)
        return ArrayAnalysis.AnalyzeResult(shape=shape)
    elif expr.attr in ('real', 'imag') and self._isarray(expr.value.name):
        return ArrayAnalysis.AnalyzeResult(shape=expr.value)
    elif self._isarray(lhs.name):
        canonical_value = get_canonical_alias(expr.value.name, self.alias_map)
        if (canonical_value, expr.attr) in self.object_attrs:
            return ArrayAnalysis.AnalyzeResult(shape=self.object_attrs[canonical_value, expr.attr])
        else:
            typ = self.typemap[lhs.name]
            post = []
            shape = self._gen_shape_call(equiv_set, lhs, typ.ndim, None, post)
            self.object_attrs[canonical_value, expr.attr] = shape
            return ArrayAnalysis.AnalyzeResult(shape=shape, post=post)
    return None