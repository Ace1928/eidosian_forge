from antlr4 import *
from io import StringIO
import sys
def IN(self):
    return self.getToken(fugue_sqlParser.IN, 0)