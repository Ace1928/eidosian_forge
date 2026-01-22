import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_dot(self, scope, equiv_set, loc, args, kws):
    n = len(args)
    assert n >= 2
    loc = args[0].loc
    require(all([self._isarray(x.name) for x in args]))
    typs = [self.typemap[x.name] for x in args]
    dims = [ty.ndim for ty in typs]
    require(all((x > 0 for x in dims)))
    if dims[0] == 1 and dims[1] == 1:
        return None
    shapes = [equiv_set._get_shape(x) for x in args]
    if dims[0] == 1:
        asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][0], shapes[1][-2]])
        return ArrayAnalysis.AnalyzeResult(shape=tuple(shapes[1][0:-2] + shapes[1][-1:]), pre=asserts)
    if dims[1] == 1:
        asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][-1], shapes[1][0]])
        return ArrayAnalysis.AnalyzeResult(shape=tuple(shapes[0][0:-1]), pre=asserts)
    if dims[0] == 2 and dims[1] == 2:
        asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][1], shapes[1][0]])
        return ArrayAnalysis.AnalyzeResult(shape=(shapes[0][0], shapes[1][1]), pre=asserts)
    if dims[0] > 2:
        pass
    return None