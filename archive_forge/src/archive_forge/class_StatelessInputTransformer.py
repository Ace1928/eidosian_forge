import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
class StatelessInputTransformer(InputTransformer):
    """Wrapper for a stateless input transformer implemented as a function."""

    def __init__(self, func):
        self.func = func

    def __repr__(self):
        return 'StatelessInputTransformer(func={0!r})'.format(self.func)

    def push(self, line):
        """Send a line of input to the transformer, returning the
        transformed input."""
        return self.func(line)

    def reset(self):
        """No-op - exists for compatibility."""
        pass