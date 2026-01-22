import formatter
import string
from types import *
import htmllib
import piddle
def OutputLine(self, text, linebreak=0):
    if text:
        if TRACE:
            print('olt:', text)
        if TRACE:
            print('olf:', self.font.size, self.font.bold, self.font.italic, self.font.underline, self.font.face)
        self.pc.drawString(text, self.x, self.y, self.font, self.color)
        self.lineHeight = max(self.lineHeight, self.pc.fontHeight(self.font))
        self.x = self.x + self.pc.stringWidth(text, self.font)
    if linebreak:
        self.send_line_break()