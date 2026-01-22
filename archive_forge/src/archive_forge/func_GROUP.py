from antlr4 import *
from io import StringIO
import sys
def GROUP(self):
    return self.getToken(fugue_sqlParser.GROUP, 0)