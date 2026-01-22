from antlr4 import *
from io import StringIO
import sys
def R_ANGLE(self):
    return self.getToken(LaTeXParser.R_ANGLE, 0)