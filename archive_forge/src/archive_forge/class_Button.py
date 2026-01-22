import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class Button(Widget):

    def __init__(self, label, onPress):
        Widget.__init__(self)
        self.label = label
        self.onPress = onPress

    def sizeHint(self):
        return (len(self.label), 1)

    def characterReceived(self, keyID, modifier):
        if keyID == b'\r':
            self.onPress()

    def render(self, width, height, terminal):
        terminal.cursorPosition(0, 0)
        if self.focused:
            terminal.write(b'\x1b[1m' + self.label + b'\x1b[0m')
        else:
            terminal.write(self.label)