from antlr4 import *
from io import StringIO
import sys
def BETWEEN(self):
    return self.getToken(fugue_sqlParser.BETWEEN, 0)