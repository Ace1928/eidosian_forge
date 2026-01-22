from antlr4 import *
from io import StringIO
import sys
def DROP(self):
    return self.getToken(fugue_sqlParser.DROP, 0)