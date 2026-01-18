import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def to_shape(typ, index, dsize):
    if isinstance(typ, types.SliceType):
        return self.slice_size(index, dsize, equiv_set, scope, stmts)
    elif isinstance(typ, types.Number):
        return (None, None)
    else:
        require(False)