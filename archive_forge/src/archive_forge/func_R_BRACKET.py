from antlr4 import *
from io import StringIO
import sys
def R_BRACKET(self):
    return self.getToken(LaTeXParser.R_BRACKET, 0)