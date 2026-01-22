from antlr4 import *
from io import StringIO
import sys
def OVER(self):
    return self.getToken(fugue_sqlParser.OVER, 0)