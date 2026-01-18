from pyparsing import *
from sys import stdin, argv, exit
def while_begin_action(self, text, loc, arg):
    """Code executed after recognising a while statement (while keyword)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('WHILE_BEGIN:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.false_label_number += 1
    self.label_number = self.false_label_number
    self.codegen.newline_label('while{0}'.format(self.label_number), True, True)