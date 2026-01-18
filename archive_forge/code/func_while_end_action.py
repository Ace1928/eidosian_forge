from pyparsing import *
from sys import stdin, argv, exit
def while_end_action(self, text, loc, arg):
    """Code executed after recognising a whole while statement"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('WHILE_END:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.label_number = self.label_stack.pop()
    label = self.codegen.label('while{0}'.format(self.label_number), True, False)
    self.codegen.unconditional_jump(label)
    self.codegen.newline_label('false{0}'.format(self.label_stack.pop()), True, True)
    self.codegen.newline_label('exit{0}'.format(self.label_number), True, True)