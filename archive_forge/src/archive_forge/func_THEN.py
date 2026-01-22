from antlr4 import *
from io import StringIO
import sys
def THEN(self):
    return self.getToken(fugue_sqlParser.THEN, 0)