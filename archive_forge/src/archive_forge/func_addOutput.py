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
def addOutput(self, data, isAsync=None, **kwargs):
    isAsync = _get_async_param(isAsync, **kwargs)
    if isAsync:
        self.terminal.eraseLine()
        self.terminal.cursorBackward(len(self.lineBuffer) + len(self.ps[self.pn]))
    self.terminal.write(data)
    if isAsync:
        if self._needsNewline():
            self.terminal.nextLine()
        self.terminal.write(self.ps[self.pn])
        if self.lineBuffer:
            oldBuffer = self.lineBuffer
            self.lineBuffer = []
            self.lineBufferIndex = 0
            self._deliverBuffer(oldBuffer)