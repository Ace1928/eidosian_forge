from antlr4 import *
from io import StringIO
import sys
def ADD(self):
    return self.getToken(LaTeXParser.ADD, 0)