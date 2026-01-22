from antlr4 import *
from io import StringIO
import sys
def FUNC_INT(self):
    return self.getToken(LaTeXParser.FUNC_INT, 0)