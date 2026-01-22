from antlr4 import *
from io import StringIO
import sys
def AFTER(self):
    return self.getToken(fugue_sqlParser.AFTER, 0)