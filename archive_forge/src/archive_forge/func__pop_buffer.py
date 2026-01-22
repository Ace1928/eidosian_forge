import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _pop_buffer(self):
    """pop the most recent capturing buffer from this Context."""
    return self._buffer_stack.pop()