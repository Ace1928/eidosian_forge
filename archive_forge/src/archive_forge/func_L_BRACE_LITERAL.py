from antlr4 import *
from io import StringIO
import sys
def L_BRACE_LITERAL(self):
    return self.getToken(LaTeXParser.L_BRACE_LITERAL, 0)