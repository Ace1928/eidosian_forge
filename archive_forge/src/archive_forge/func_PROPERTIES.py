from antlr4 import *
from io import StringIO
import sys
def PROPERTIES(self):
    return self.getToken(fugue_sqlParser.PROPERTIES, 0)