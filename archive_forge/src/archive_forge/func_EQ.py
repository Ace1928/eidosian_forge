from antlr4 import *
from io import StringIO
import sys
def EQ(self):
    return self.getToken(sqlParser.EQ, 0)