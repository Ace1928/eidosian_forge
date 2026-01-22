from antlr4 import *
from io import StringIO
import sys
def ALTER(self):
    return self.getToken(fugue_sqlParser.ALTER, 0)