from antlr4 import *
from io import StringIO
import sys
def SELECT(self):
    return self.getToken(fugue_sqlParser.SELECT, 0)