from antlr4 import *
from io import StringIO
import sys
def IGNORE(self):
    return self.getToken(fugue_sqlParser.IGNORE, 0)