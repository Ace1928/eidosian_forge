from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
class ControlSequenceParser:

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
    for ch, fName in (('A', 'Up'), ('B', 'Down'), ('C', 'Forward'), ('D', 'Backward')):
        exec(ch + ' = _makeSimple(ch, fName)')
    del _makeSimple

    def h(self, proto, handler, buf):
        try:
            modes = [int(mode) for mode in buf.split(b';')]
        except ValueError:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'h')
        else:
            handler.setModes(modes)

    def l(self, proto, handler, buf):
        try:
            modes = [int(mode) for mode in buf.split(b';')]
        except ValueError:
            handler.unhandledControlSequence(b'\x1b[' + buf + 'l')
        else:
            handler.resetModes(modes)

    def r(self, proto, handler, buf):
        parts = buf.split(b';')
        if len(parts) == 1:
            handler.setScrollRegion(None, None)
        elif len(parts) == 2:
            try:
                if parts[0]:
                    pt = int(parts[0])
                else:
                    pt = None
                if parts[1]:
                    pb = int(parts[1])
                else:
                    pb = None
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'r')
            else:
                handler.setScrollRegion(pt, pb)
        else:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'r')

    def K(self, proto, handler, buf):
        if not buf:
            handler.eraseToLineEnd()
        elif buf == b'1':
            handler.eraseToLineBeginning()
        elif buf == b'2':
            handler.eraseLine()
        else:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'K')

    def H(self, proto, handler, buf):
        handler.cursorHome()

    def J(self, proto, handler, buf):
        if not buf:
            handler.eraseToDisplayEnd()
        elif buf == b'1':
            handler.eraseToDisplayBeginning()
        elif buf == b'2':
            handler.eraseDisplay()
        else:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'J')

    def P(self, proto, handler, buf):
        if not buf:
            handler.deleteCharacter(1)
        else:
            try:
                n = int(buf)
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'P')
            else:
                handler.deleteCharacter(n)

    def L(self, proto, handler, buf):
        if not buf:
            handler.insertLine(1)
        else:
            try:
                n = int(buf)
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'L')
            else:
                handler.insertLine(n)

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

    def n(self, proto, handler, buf):
        if buf == b'6':
            x, y = handler.reportCursorPosition()
            proto.transport.write(b'\x1b[%d;%dR' % (x + 1, y + 1))
        else:
            handler.unhandledControlSequence(b'\x1b[' + buf + b'n')

    def m(self, proto, handler, buf):
        if not buf:
            handler.selectGraphicRendition(NORMAL)
        else:
            attrs = []
            for a in buf.split(b';'):
                try:
                    a = int(a)
                except ValueError:
                    pass
                attrs.append(a)
            handler.selectGraphicRendition(*attrs)