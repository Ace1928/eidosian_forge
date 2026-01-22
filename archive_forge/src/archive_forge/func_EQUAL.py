from antlr4 import *
from io import StringIO
import sys
def EQUAL(self):
    return self.getToken(LaTeXParser.EQUAL, 0)