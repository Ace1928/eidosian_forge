import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
def _needsNewline(self):
    w = self.terminal.lastWrite
    return not w.endswith(b'\n') and (not w.endswith(b'\x1bE'))