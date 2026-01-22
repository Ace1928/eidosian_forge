from antlr4 import *
from io import StringIO
import sys
def CONSTRAINT(self):
    return self.getToken(fugue_sqlParser.CONSTRAINT, 0)