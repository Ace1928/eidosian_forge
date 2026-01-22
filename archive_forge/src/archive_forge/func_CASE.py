from antlr4 import *
from io import StringIO
import sys
def CASE(self):
    return self.getToken(fugue_sqlParser.CASE, 0)