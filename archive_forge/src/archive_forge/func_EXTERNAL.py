from antlr4 import *
from io import StringIO
import sys
def EXTERNAL(self):
    return self.getToken(fugue_sqlParser.EXTERNAL, 0)