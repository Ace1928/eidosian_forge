from pyparsing import *
from sys import stdin, argv, exit
def andexp_action(self, text, loc, arg):
    """Code executed after recognising a andexp expression (something and something)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('AND+EXP:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    label = self.codegen.label('false{0}'.format(self.false_label_number), True, False)
    self.codegen.jump(self.relexp_code, True, label)
    self.andexp_code = self.relexp_code
    return self.andexp_code