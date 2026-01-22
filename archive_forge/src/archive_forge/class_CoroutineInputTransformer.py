import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
class CoroutineInputTransformer(InputTransformer):
    """Wrapper for an input transformer implemented as a coroutine."""

    def __init__(self, coro, **kwargs):
        self.coro = coro(**kwargs)
        next(self.coro)

    def __repr__(self):
        return 'CoroutineInputTransformer(coro={0!r})'.format(self.coro)

    def push(self, line):
        """Send a line of input to the transformer, returning the
        transformed input or None if the transformer is waiting for more
        input.
        """
        return self.coro.send(line)

    def reset(self):
        """Return, transformed any lines that the transformer has
        accumulated, and reset its internal state.
        """
        return self.coro.send(None)