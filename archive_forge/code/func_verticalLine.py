import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def verticalLine(terminal, x, top, bottom):
    terminal.selectCharacterSet(insults.CS_DRAWING, insults.G0)
    for n in range(top, bottom):
        terminal.cursorPosition(x, n)
        terminal.write(b'x')
    terminal.selectCharacterSet(insults.CS_US, insults.G0)