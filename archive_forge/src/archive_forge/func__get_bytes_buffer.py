import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def _get_bytes_buffer(ptr, nbytes):
    """
    Get a ctypes array of *nbytes* starting at *ptr*.
    """
    if isinstance(ptr, ctypes.c_void_p):
        ptr = ptr.value
    arrty = ctypes.c_byte * nbytes
    return arrty.from_address(ptr)