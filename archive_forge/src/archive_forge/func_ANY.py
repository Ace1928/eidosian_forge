from antlr4 import *
from io import StringIO
import sys
def ANY(self):
    return self.getToken(fugue_sqlParser.ANY, 0)