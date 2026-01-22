from antlr4 import *
from io import StringIO
import sys
def AS(self):
    return self.getToken(fugue_sqlParser.AS, 0)