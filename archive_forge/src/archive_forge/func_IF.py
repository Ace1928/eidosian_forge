from antlr4 import *
from io import StringIO
import sys
def IF(self):
    return self.getToken(fugue_sqlParser.IF, 0)