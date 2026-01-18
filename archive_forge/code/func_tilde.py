from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def tilde(self, proto, handler, buf):
    map = {1: proto.HOME, 2: proto.INSERT, 3: proto.DELETE, 4: proto.END, 5: proto.PGUP, 6: proto.PGDN, 15: proto.F5, 17: proto.F6, 18: proto.F7, 19: proto.F8, 20: proto.F9, 21: proto.F10, 23: proto.F11, 24: proto.F12}
    if buf.startswith(b'\x1b['):
        ch = buf[2:]
        try:
            v = int(ch)
        except ValueError:
            handler.unhandledControlSequence(buf + b'~')
        else:
            symbolic = map.get(v)
            if symbolic is not None:
                handler.keystrokeReceived(map[v], None)
            else:
                handler.unhandledControlSequence(buf + b'~')
    else:
        handler.unhandledControlSequence(buf + b'~')