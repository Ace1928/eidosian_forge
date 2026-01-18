import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def unpack_shapes(a, aty):
    if isinstance(aty, types.ArrayCompatible):
        ary = context.make_array(aty)(context, builder, a)
        return cgutils.unpack_tuple(builder, ary.shape)
    elif isinstance(aty, types.BaseTuple):
        return cgutils.unpack_tuple(builder, a)
    else:
        return [a]