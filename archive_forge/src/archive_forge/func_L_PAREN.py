from antlr4 import *
from io import StringIO
import sys
def L_PAREN(self):
    return self.getToken(LaTeXParser.L_PAREN, 0)