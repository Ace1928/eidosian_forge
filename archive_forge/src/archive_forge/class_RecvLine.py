import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
class RecvLine(insults.TerminalProtocol):
    """
    L{TerminalProtocol} which adds line editing features.

    Clients will be prompted for lines of input with all the usual
    features: character echoing, left and right arrow support for
    moving the cursor to different areas of the line buffer, backspace
    and delete for removing characters, and insert for toggling
    between typeover and insert mode.  Tabs will be expanded to enough
    spaces to move the cursor to the next tabstop (every four
    characters by default).  Enter causes the line buffer to be
    cleared and the line to be passed to the lineReceived() method
    which, by default, does nothing.  Subclasses are responsible for
    redrawing the input prompt (this will probably change).
    """
    width = 80
    height = 24
    TABSTOP = 4
    ps = (b'>>> ', b'... ')
    pn = 0
    _printableChars = string.printable.encode('ascii')
    _log = Logger()

    def connectionMade(self):
        self.lineBuffer = []
        self.lineBufferIndex = 0
        t = self.terminal
        self.keyHandlers = {t.LEFT_ARROW: self.handle_LEFT, t.RIGHT_ARROW: self.handle_RIGHT, t.TAB: self.handle_TAB, b'\r': self.handle_RETURN, b'\n': self.handle_RETURN, t.BACKSPACE: self.handle_BACKSPACE, t.DELETE: self.handle_DELETE, t.INSERT: self.handle_INSERT, t.HOME: self.handle_HOME, t.END: self.handle_END}
        self.initializeScreen()

    def initializeScreen(self):
        self.terminal.reset()
        self.terminal.write(self.ps[self.pn])
        self.setInsertMode()

    def currentLineBuffer(self):
        s = b''.join(self.lineBuffer)
        return (s[:self.lineBufferIndex], s[self.lineBufferIndex:])

    def setInsertMode(self):
        self.mode = 'insert'
        self.terminal.setModes([insults.modes.IRM])

    def setTypeoverMode(self):
        self.mode = 'typeover'
        self.terminal.resetModes([insults.modes.IRM])

    def drawInputLine(self):
        """
        Write a line containing the current input prompt and the current line
        buffer at the current cursor position.
        """
        self.terminal.write(self.ps[self.pn] + b''.join(self.lineBuffer))

    def terminalSize(self, width, height):
        self.terminal.eraseDisplay()
        self.terminal.cursorHome()
        self.width = width
        self.height = height
        self.drawInputLine()

    def unhandledControlSequence(self, seq):
        pass

    def keystrokeReceived(self, keyID, modifier):
        m = self.keyHandlers.get(keyID)
        if m is not None:
            m()
        elif keyID in self._printableChars:
            self.characterReceived(keyID, False)
        else:
            self._log.warn('Received unhandled keyID: {keyID!r}', keyID=keyID)

    def characterReceived(self, ch, moreCharactersComing):
        if self.mode == 'insert':
            self.lineBuffer.insert(self.lineBufferIndex, ch)
        else:
            self.lineBuffer[self.lineBufferIndex:self.lineBufferIndex + 1] = [ch]
        self.lineBufferIndex += 1
        self.terminal.write(ch)

    def handle_TAB(self):
        n = self.TABSTOP - len(self.lineBuffer) % self.TABSTOP
        self.terminal.cursorForward(n)
        self.lineBufferIndex += n
        self.lineBuffer.extend(iterbytes(b' ' * n))

    def handle_LEFT(self):
        if self.lineBufferIndex > 0:
            self.lineBufferIndex -= 1
            self.terminal.cursorBackward()

    def handle_RIGHT(self):
        if self.lineBufferIndex < len(self.lineBuffer):
            self.lineBufferIndex += 1
            self.terminal.cursorForward()

    def handle_HOME(self):
        if self.lineBufferIndex:
            self.terminal.cursorBackward(self.lineBufferIndex)
            self.lineBufferIndex = 0

    def handle_END(self):
        offset = len(self.lineBuffer) - self.lineBufferIndex
        if offset:
            self.terminal.cursorForward(offset)
            self.lineBufferIndex = len(self.lineBuffer)

    def handle_BACKSPACE(self):
        if self.lineBufferIndex > 0:
            self.lineBufferIndex -= 1
            del self.lineBuffer[self.lineBufferIndex]
            self.terminal.cursorBackward()
            self.terminal.deleteCharacter()

    def handle_DELETE(self):
        if self.lineBufferIndex < len(self.lineBuffer):
            del self.lineBuffer[self.lineBufferIndex]
            self.terminal.deleteCharacter()

    def handle_RETURN(self):
        line = b''.join(self.lineBuffer)
        self.lineBuffer = []
        self.lineBufferIndex = 0
        self.terminal.nextLine()
        self.lineReceived(line)

    def handle_INSERT(self):
        assert self.mode in ('typeover', 'insert')
        if self.mode == 'typeover':
            self.setInsertMode()
        else:
            self.setTypeoverMode()

    def lineReceived(self, line):
        pass