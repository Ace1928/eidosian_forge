from antlr4 import *
from io import StringIO
import sys
def WITH(self):
    return self.getToken(fugue_sqlParser.WITH, 0)