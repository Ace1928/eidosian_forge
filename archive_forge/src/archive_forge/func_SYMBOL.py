from antlr4 import *
from io import StringIO
import sys
def SYMBOL(self):
    return self.getToken(LaTeXParser.SYMBOL, 0)