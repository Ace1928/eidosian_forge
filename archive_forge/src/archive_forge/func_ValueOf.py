import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def ValueOf(self, token_type):
    return self.valuesDict.get(token_type, -1)