from antlr4 import *
from io import StringIO
import sys
def TRIM(self):
    return self.getToken(fugue_sqlParser.TRIM, 0)