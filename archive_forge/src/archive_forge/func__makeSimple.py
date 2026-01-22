from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def _makeSimple(ch, fName):
    n = 'cursor' + fName

    def simple(self, proto, handler, buf):
        if not buf:
            getattr(handler, n)(1)
        else:
            try:
                m = int(buf)
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + ch)
            else:
                getattr(handler, n)(m)
    return simple