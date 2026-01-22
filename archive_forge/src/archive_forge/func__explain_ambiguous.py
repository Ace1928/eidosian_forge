import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def _explain_ambiguous(self, *args, **kws):
    """
        Callback for the C _Dispatcher object.
        """
    assert not kws, 'kwargs not handled'
    args = tuple([self.typeof_pyval(a) for a in args])
    sigs = self.nopython_signatures
    self.typingctx.resolve_overload(self.py_func, sigs, args, kws, allow_ambiguous=False)