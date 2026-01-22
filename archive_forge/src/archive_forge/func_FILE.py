from antlr4 import *
from io import StringIO
import sys
def FILE(self):
    return self.getToken(fugue_sqlParser.FILE, 0)