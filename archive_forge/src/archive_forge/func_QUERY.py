from antlr4 import *
from io import StringIO
import sys
def QUERY(self):
    return self.getToken(fugue_sqlParser.QUERY, 0)