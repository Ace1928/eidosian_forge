from antlr4 import *
from io import StringIO
import sys
def CMD_MATHIT(self):
    return self.getToken(LaTeXParser.CMD_MATHIT, 0)