from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def _resolve_user_function_type(self, func, args, kws, literals=None):
    functy = self._lookup_global(func)
    if functy is not None:
        func = functy
    if isinstance(func, types.Type):
        func_type = self.resolve_getattr(func, '__call__')
        if func_type is not None:
            return self.resolve_function_type(func_type, args, kws)
    if isinstance(func, types.Callable):
        return func.get_call_type(self, args, kws)