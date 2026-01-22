from antlr4 import *
from io import StringIO
import sys
def STRING(self):
    return self.getToken(fugue_sqlParser.STRING, 0)