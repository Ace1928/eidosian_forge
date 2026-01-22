import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class Border(Widget):

    def __init__(self, containee):
        Widget.__init__(self)
        self.containee = containee
        self.containee.parent = self

    def focusReceived(self):
        return self.containee.focusReceived()

    def focusLost(self):
        return self.containee.focusLost()

    def keystrokeReceived(self, keyID, modifier):
        return self.containee.keystrokeReceived(keyID, modifier)

    def sizeHint(self):
        hint = self.containee.sizeHint()
        if hint is None:
            hint = (None, None)
        if hint[0] is None:
            x = None
        else:
            x = hint[0] + 2
        if hint[1] is None:
            y = None
        else:
            y = hint[1] + 2
        return (x, y)

    def filthy(self):
        self.containee.filthy()
        Widget.filthy(self)

    def render(self, width, height, terminal):
        if self.containee.focused:
            terminal.write(b'\x1b[31m')
        rectangle(terminal, (0, 0), (width, height))
        terminal.write(b'\x1b[0m')
        wrap = BoundedTerminalWrapper(terminal, width - 2, height - 2, 1, 1)
        self.containee.draw(width - 2, height - 2, wrap)