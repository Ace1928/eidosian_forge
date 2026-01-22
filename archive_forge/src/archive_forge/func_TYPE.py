from antlr4 import *
from io import StringIO
import sys
def TYPE(self):
    return self.getToken(fugue_sqlParser.TYPE, 0)