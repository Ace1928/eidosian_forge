from antlr4 import *
from io import StringIO
import sys
def END(self):
    return self.getToken(fugue_sqlParser.END, 0)