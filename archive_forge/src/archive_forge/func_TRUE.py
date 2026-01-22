from antlr4 import *
from io import StringIO
import sys
def TRUE(self):
    return self.getToken(fugue_sqlParser.TRUE, 0)