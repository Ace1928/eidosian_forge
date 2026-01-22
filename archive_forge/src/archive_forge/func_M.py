from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def M(self, proto, handler, buf):
    if not buf:
        handler.deleteLine(1)
    else:
        try:
            n = int(buf)
        except ValueError:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'M')
        else:
            handler.deleteLine(n)