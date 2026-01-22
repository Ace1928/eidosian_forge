from antlr4 import *
from io import StringIO
import sys
def CARET(self):
    return self.getToken(LaTeXParser.CARET, 0)