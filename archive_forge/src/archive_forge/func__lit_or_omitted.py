from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def _lit_or_omitted(value):
    """Returns a Literal instance if the type of value is supported;
    otherwise, return `Omitted(value)`.
    """
    try:
        return types.literal(value)
    except LiteralTypingError:
        return types.Omitted(value)