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
class Manhole(recvline.HistoricRecvLine):
    """
    Mediator between a fancy line source and an interactive interpreter.

    This accepts lines from its transport and passes them on to a
    L{ManholeInterpreter}.  Control commands (^C, ^D, ^\\) are also handled
    with something approximating their normal terminal-mode behavior.  It
    can optionally be constructed with a dict which will be used as the
    local namespace for any code executed.
    """
    namespace = None

    def __init__(self, namespace=None):
        recvline.HistoricRecvLine.__init__(self)
        if namespace is not None:
            self.namespace = namespace.copy()

    def connectionMade(self):
        recvline.HistoricRecvLine.connectionMade(self)
        self.interpreter = ManholeInterpreter(self, self.namespace)
        self.keyHandlers[CTRL_C] = self.handle_INT
        self.keyHandlers[CTRL_D] = self.handle_EOF
        self.keyHandlers[CTRL_L] = self.handle_FF
        self.keyHandlers[CTRL_A] = self.handle_HOME
        self.keyHandlers[CTRL_E] = self.handle_END
        self.keyHandlers[CTRL_BACKSLASH] = self.handle_QUIT

    def handle_INT(self):
        """
        Handle ^C as an interrupt keystroke by resetting the current input
        variables to their initial state.
        """
        self.pn = 0
        self.lineBuffer = []
        self.lineBufferIndex = 0
        self.interpreter.resetBuffer()
        self.terminal.nextLine()
        self.terminal.write(b'KeyboardInterrupt')
        self.terminal.nextLine()
        self.terminal.write(self.ps[self.pn])

    def handle_EOF(self):
        if self.lineBuffer:
            self.terminal.write(b'\x07')
        else:
            self.handle_QUIT()

    def handle_FF(self):
        """
        Handle a 'form feed' byte - generally used to request a screen
        refresh/redraw.
        """
        self.terminal.eraseDisplay()
        self.terminal.cursorHome()
        self.drawInputLine()

    def handle_QUIT(self):
        self.terminal.loseConnection()

    def _needsNewline(self):
        w = self.terminal.lastWrite
        return not w.endswith(b'\n') and (not w.endswith(b'\x1bE'))

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

    def lineReceived(self, line):
        more = self.interpreter.push(line)
        self.pn = bool(more)
        if self._needsNewline():
            self.terminal.nextLine()
        self.terminal.write(self.ps[self.pn])