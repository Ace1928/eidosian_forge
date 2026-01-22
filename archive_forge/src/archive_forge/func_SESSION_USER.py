from antlr4 import *
from io import StringIO
import sys
def SESSION_USER(self):
    return self.getToken(fugue_sqlParser.SESSION_USER, 0)