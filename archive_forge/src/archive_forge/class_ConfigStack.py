import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
class ConfigStack:
    """A stack for tracking target configurations in the compiler.

    It stores the stack in a thread-local class attribute. All instances in the
    same thread will see the same stack.
    """

    @classmethod
    def top_or_none(cls):
        """Get the TOS or return None if no config is set.
        """
        self = cls()
        if self:
            flags = self.top()
        else:
            flags = None
        return flags

    def __init__(self):
        self._stk = _FlagsStack()

    def top(self):
        return self._stk.top()

    def __len__(self):
        return len(self._stk)

    def enter(self, flags):
        """Returns a contextmanager that performs ``push(flags)`` on enter and
        ``pop()`` on exit.
        """
        return self._stk.enter(flags)