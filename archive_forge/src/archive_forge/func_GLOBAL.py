from antlr4 import *
from io import StringIO
import sys
def GLOBAL(self):
    return self.getToken(fugue_sqlParser.GLOBAL, 0)