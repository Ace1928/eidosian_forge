from antlr4 import *
from io import StringIO
import sys
def FROM(self):
    return self.getToken(fugue_sqlParser.FROM, 0)