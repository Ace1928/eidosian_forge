from antlr4 import *
from io import StringIO
import sys
def DEFINED(self):
    return self.getToken(fugue_sqlParser.DEFINED, 0)