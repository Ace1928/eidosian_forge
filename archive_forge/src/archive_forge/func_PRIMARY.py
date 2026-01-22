from antlr4 import *
from io import StringIO
import sys
def PRIMARY(self):
    return self.getToken(fugue_sqlParser.PRIMARY, 0)