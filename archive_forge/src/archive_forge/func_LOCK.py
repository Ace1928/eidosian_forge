from antlr4 import *
from io import StringIO
import sys
def LOCK(self):
    return self.getToken(fugue_sqlParser.LOCK, 0)