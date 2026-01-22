import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def _expand_integer(ty):
    """
    If *ty* is an integer, expand it to a machine int (like Numpy).
    """
    if isinstance(ty, types.Integer):
        if ty.signed:
            return max(types.intp, ty)
        else:
            return max(types.uintp, ty)
    elif isinstance(ty, types.Boolean):
        return types.intp
    else:
        return ty