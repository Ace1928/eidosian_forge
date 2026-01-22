from antlr4 import *
from io import StringIO
import sys
class CeilContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.val = None

    def L_CEIL(self):
        return self.getToken(LaTeXParser.L_CEIL, 0)

    def R_CEIL(self):
        return self.getToken(LaTeXParser.R_CEIL, 0)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_ceil