from antlr4 import *
from io import StringIO
import sys
def Variables(self):
    return self.getToken(AutolevParser.Variables, 0)