from antlr4 import *
from io import StringIO
import sys
def CMD_CDOT(self):
    return self.getToken(LaTeXParser.CMD_CDOT, 0)