import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def union_defs(self, defs, redefined):
    """Union with the given defs dictionary. This is meant to handle
        branch join-point, where a variable may have been defined in more
        than one branches.
        """
    for k, v in defs.items():
        if v > 0:
            self.define(k, redefined)