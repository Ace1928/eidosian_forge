from antlr4 import *
from io import StringIO
import sys
def Points(self):
    return self.getToken(AutolevParser.Points, 0)