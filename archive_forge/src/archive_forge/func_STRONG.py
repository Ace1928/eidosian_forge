from antlr4 import *
from io import StringIO
import sys
def STRONG(self):
    return self.getToken(fugue_sqlParser.STRONG, 0)