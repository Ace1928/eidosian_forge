from antlr4 import *
from io import StringIO
import sys
class EqualityContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.ExprContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.ExprContext, i)

    def EQUAL(self):
        return self.getToken(LaTeXParser.EQUAL, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_equality