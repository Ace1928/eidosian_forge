from antlr4 import *
from io import StringIO
import sys
def DESCRIBE(self):
    return self.getToken(fugue_sqlParser.DESCRIBE, 0)