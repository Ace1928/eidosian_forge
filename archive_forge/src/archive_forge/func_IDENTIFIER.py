from antlr4 import *
from io import StringIO
import sys
def IDENTIFIER(self):
    return self.getToken(fugue_sqlParser.IDENTIFIER, 0)