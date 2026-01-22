from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def _handleControlSequence(self, buf, terminal):
    f = getattr(self.controlSequenceParser, CST.get(terminal, terminal).decode('ascii'), None)
    if f is None:
        self.terminal.unhandledControlSequence(b'\x1b[' + buf + terminal)
    else:
        f(self, self.terminal, buf)