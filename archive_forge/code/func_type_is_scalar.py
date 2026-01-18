import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def type_is_scalar(typ):
    """ Returns True if the type of 'typ' is a scalar type, according to
    NumPy rules. False otherwise.
    https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
    """
    ok = (types.Boolean, types.Number, types.UnicodeType, types.StringLiteral, types.NPTimedelta, types.NPDatetime)
    return isinstance(typ, ok)