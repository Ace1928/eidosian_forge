import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
class HistoricRecvLine(RecvLine):
    """
    L{TerminalProtocol} which adds both basic line-editing features and input history.

    Everything supported by L{RecvLine} is also supported by this class.  In addition, the
    up and down arrows traverse the input history.  Each received line is automatically
    added to the end of the input history.
    """

    def connectionMade(self):
        RecvLine.connectionMade(self)
        self.historyLines = []
        self.historyPosition = 0
        t = self.terminal
        self.keyHandlers.update({t.UP_ARROW: self.handle_UP, t.DOWN_ARROW: self.handle_DOWN})

    def currentHistoryBuffer(self):
        b = tuple(self.historyLines)
        return (b[:self.historyPosition], b[self.historyPosition:])

    def _deliverBuffer(self, buf):
        if buf:
            for ch in iterbytes(buf[:-1]):
                self.characterReceived(ch, True)
            self.characterReceived(buf[-1:], False)

    def handle_UP(self):
        if self.lineBuffer and self.historyPosition == len(self.historyLines):
            self.historyLines.append(b''.join(self.lineBuffer))
        if self.historyPosition > 0:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition -= 1
            self.lineBuffer = []
            self._deliverBuffer(self.historyLines[self.historyPosition])

    def handle_DOWN(self):
        if self.historyPosition < len(self.historyLines) - 1:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition += 1
            self.lineBuffer = []
            self._deliverBuffer(self.historyLines[self.historyPosition])
        else:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition = len(self.historyLines)
            self.lineBuffer = []
            self.lineBufferIndex = 0

    def handle_RETURN(self):
        if self.lineBuffer:
            self.historyLines.append(b''.join(self.lineBuffer))
        self.historyPosition = len(self.historyLines)
        return RecvLine.handle_RETURN(self)