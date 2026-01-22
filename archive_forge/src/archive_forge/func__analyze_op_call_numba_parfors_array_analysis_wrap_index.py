import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numba_parfors_array_analysis_wrap_index(self, scope, equiv_set, loc, args, kws):
    """ Analyze wrap_index calls added by a previous run of
            Array Analysis
        """
    require(len(args) == 2)
    slice_size = args[0].name
    dim_size = args[1].name
    slice_eq = equiv_set._get_or_add_ind(slice_size)
    dim_eq = equiv_set._get_or_add_ind(dim_size)
    if (slice_eq, dim_eq) in equiv_set.wrap_map:
        wrap_ind = equiv_set.wrap_map[slice_eq, dim_eq]
        require(wrap_ind in equiv_set.ind_to_var)
        vs = equiv_set.ind_to_var[wrap_ind]
        require(vs != [])
        return ArrayAnalysis.AnalyzeResult(shape=(vs[0],))
    else:
        return ArrayAnalysis.AnalyzeResult(shape=WrapIndexMeta(slice_eq, dim_eq))