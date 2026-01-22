from antlr4 import *
from io import StringIO
import sys
def STRUCT(self):
    return self.getToken(fugue_sqlParser.STRUCT, 0)