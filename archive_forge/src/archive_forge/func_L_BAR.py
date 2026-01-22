from antlr4 import *
from io import StringIO
import sys
def L_BAR(self):
    return self.getToken(LaTeXParser.L_BAR, 0)