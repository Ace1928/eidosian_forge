from antlr4 import *
from io import StringIO
import sys
def additive_sempred(self, localctx: AdditiveContext, predIndex: int):
    if predIndex == 1:
        return self.precpred(self._ctx, 2)