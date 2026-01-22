from antlr4 import *
from io import StringIO
import sys
def INTEGER_VALUE(self):
    return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)