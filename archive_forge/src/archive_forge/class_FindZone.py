import itertools
import re
from tkinter import SEL_FIRST, SEL_LAST, Frame, Label, PhotoImage, Scrollbar, Text, Tk
class FindZone(Zone):

    def addTags(self, m):
        color = next(self.colorCycle)
        self.txt.tag_add(color, '1.0+%sc' % m.start(), '1.0+%sc' % m.end())
        try:
            self.txt.tag_add('emph' + color, '1.0+%sc' % m.start('emph'), '1.0+%sc' % m.end('emph'))
        except:
            pass

    def substitute(self, *args):
        for color in colors:
            self.txt.tag_remove(color, '1.0', 'end')
            self.txt.tag_remove('emph' + color, '1.0', 'end')
        self.rex = re.compile('')
        self.rex = re.compile(self.fld.get('1.0', 'end')[:-1], re.MULTILINE)
        try:
            re.compile('(?P<emph>%s)' % self.fld.get(SEL_FIRST, SEL_LAST))
            self.rexSel = re.compile('%s(?P<emph>%s)%s' % (self.fld.get('1.0', SEL_FIRST), self.fld.get(SEL_FIRST, SEL_LAST), self.fld.get(SEL_LAST, 'end')[:-1]), re.MULTILINE)
        except:
            self.rexSel = self.rex
        self.rexSel.sub(self.addTags, self.txt.get('1.0', 'end'))