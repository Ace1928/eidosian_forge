import itertools
import re
from tkinter import SEL_FIRST, SEL_LAST, Frame, Label, PhotoImage, Scrollbar, Text, Tk
def addTags(self, m):
    s = sz.rex.sub(self.repl, m.group())
    self.txt.delete('1.0+%sc' % (m.start() + self.diff), '1.0+%sc' % (m.end() + self.diff))
    self.txt.insert('1.0+%sc' % (m.start() + self.diff), s, next(self.colorCycle))
    self.diff += len(s) - (m.end() - m.start())