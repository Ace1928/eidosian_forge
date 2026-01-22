from antlr4 import *
from io import StringIO
import sys
def PLUS(self):
    return self.getToken(fugue_sqlParser.PLUS, 0)