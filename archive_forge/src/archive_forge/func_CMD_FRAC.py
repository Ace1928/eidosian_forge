from antlr4 import *
from io import StringIO
import sys
def CMD_FRAC(self):
    return self.getToken(LaTeXParser.CMD_FRAC, 0)