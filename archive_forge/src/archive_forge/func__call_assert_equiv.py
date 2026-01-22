import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _call_assert_equiv(self, scope, loc, equiv_set, args, names=None):
    insts = self._make_assert_equiv(scope, loc, equiv_set, args, names=names)
    if len(args) > 1:
        equiv_set.insert_equiv(*args)
    return insts