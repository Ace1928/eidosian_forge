from antlr4 import *
from io import StringIO
import sys
def WHERE(self):
    return self.getToken(fugue_sqlParser.WHERE, 0)