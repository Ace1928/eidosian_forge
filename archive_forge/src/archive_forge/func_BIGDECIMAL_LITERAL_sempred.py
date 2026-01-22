from antlr4 import *
from io import StringIO
import sys
def BIGDECIMAL_LITERAL_sempred(self, localctx: RuleContext, predIndex: int):
    if predIndex == 3:
        return self.isValidDecimal