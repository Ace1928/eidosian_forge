from antlr4 import *
from io import StringIO
import sys
def DELIMITED(self):
    return self.getToken(fugue_sqlParser.DELIMITED, 0)