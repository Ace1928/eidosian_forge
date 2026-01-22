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
class CallStack(Sequence):
    """
    A compile-time call stack
    """

    def __init__(self):
        self._stack = []
        self._lock = threading.RLock()

    def __getitem__(self, index):
        """
        Returns item in the stack where index=0 is the top and index=1 is
        the second item from the top.
        """
        return self._stack[len(self) - index - 1]

    def __len__(self):
        return len(self._stack)

    @contextlib.contextmanager
    def register(self, target, typeinfer, func_id, args):
        if self.match(func_id.func, args):
            msg = 'compiler re-entrant to the same function signature'
            raise errors.NumbaRuntimeError(msg)
        self._lock.acquire()
        self._stack.append(CallFrame(target, typeinfer, func_id, args))
        try:
            yield
        finally:
            self._stack.pop()
            self._lock.release()

    def finditer(self, py_func):
        """
        Yields frame that matches the function object starting from the top
        of stack.
        """
        for frame in self:
            if frame.func_id.func is py_func:
                yield frame

    def findfirst(self, py_func):
        """
        Returns the first result from `.finditer(py_func)`; or None if no match.
        """
        try:
            return next(self.finditer(py_func))
        except StopIteration:
            return

    def match(self, py_func, args):
        """
        Returns first function that matches *py_func* and the arguments types in
        *args*; or, None if no match.
        """
        for frame in self.finditer(py_func):
            if frame.args == args:
                return frame