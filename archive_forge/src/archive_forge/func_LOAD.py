from antlr4 import *
from io import StringIO
import sys
def LOAD(self):
    return self.getToken(fugue_sqlParser.LOAD, 0)