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
class TargetConfigurationStack:
    """The target configuration stack.

    Uses the BORG pattern and stores states in threadlocal storage.

    WARNING: features associated with this class are experimental. The API
    may change without notice.
    """

    def __init__(self):
        self._stack = _RetargetStack()

    def get(self):
        """Get the current target from the top of the stack.

        May raise IndexError if the stack is empty. Users should check the size
        of the stack beforehand.
        """
        return self._stack.top()

    def __len__(self):
        """Size of the stack
        """
        return len(self._stack)

    @classmethod
    def switch_target(cls, retarget: BaseRetarget):
        """Returns a contextmanager that pushes a new retarget handler,
        an instance of `numba.core.retarget.BaseRetarget`, onto the
        target-config stack for the duration of the context-manager.
        """
        return cls()._stack.enter(retarget)