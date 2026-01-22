from antlr4 import *
from io import StringIO
import sys
def AND(self):
    return self.getToken(fugue_sqlParser.AND, 0)