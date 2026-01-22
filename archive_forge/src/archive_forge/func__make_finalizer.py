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
def _make_finalizer(self):
    """
        Return a finalizer function that will release references to
        related compiled functions.
        """
    overloads = self.overloads
    targetctx = self.targetctx

    def finalizer(shutting_down=utils.shutting_down):
        if shutting_down():
            return
        for cres in overloads.values():
            try:
                targetctx.remove_user_function(cres.entry_point)
            except KeyError:
                pass
    return finalizer