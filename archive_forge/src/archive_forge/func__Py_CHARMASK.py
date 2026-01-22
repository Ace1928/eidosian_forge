from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _Py_CHARMASK(ch):
    """
    Equivalent to the CPython macro `Py_CHARMASK()`, masks off all but the
    lowest 256 bits of ch.
    """
    return types.uint8(ch) & types.uint8(255)