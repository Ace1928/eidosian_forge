from antlr4 import *
from io import StringIO
import sys
def LIMIT(self):
    return self.getToken(fugue_sqlParser.LIMIT, 0)