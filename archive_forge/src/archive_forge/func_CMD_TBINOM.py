from antlr4 import *
from io import StringIO
import sys
def CMD_TBINOM(self):
    return self.getToken(LaTeXParser.CMD_TBINOM, 0)