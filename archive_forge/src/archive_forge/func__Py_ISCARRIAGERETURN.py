from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _Py_ISCARRIAGERETURN(ch):
    """Check if character is carriage return `\r`"""
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.CARRIAGE_RETURN