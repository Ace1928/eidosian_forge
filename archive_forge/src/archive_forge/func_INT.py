from antlr4 import *
from io import StringIO
import sys
def INT(self):
    return self.getToken(AutolevParser.INT, 0)