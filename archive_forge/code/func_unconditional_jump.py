from pyparsing import *
from sys import stdin, argv, exit
def unconditional_jump(self, label):
    """Generates an unconditional jump instruction
           label    - jump label
        """
    self.newline_text('JMP \t{0}'.format(label), True)