import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def type_can_asarray(arr):
    """ Returns True if the type of 'arr' is supported by the Numba `np.asarray`
    implementation, False otherwise.
    """
    ok = (types.Array, types.Sequence, types.Tuple, types.StringLiteral, types.Number, types.Boolean, types.containers.ListType)
    return isinstance(arr, ok)