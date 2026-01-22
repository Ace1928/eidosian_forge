from antlr4 import *
from io import StringIO
import sys
class RelationContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def relation(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.RelationContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.RelationContext, i)

    def EQUAL(self):
        return self.getToken(LaTeXParser.EQUAL, 0)

    def LT(self):
        return self.getToken(LaTeXParser.LT, 0)

    def LTE(self):
        return self.getToken(LaTeXParser.LTE, 0)

    def GT(self):
        return self.getToken(LaTeXParser.GT, 0)

    def GTE(self):
        return self.getToken(LaTeXParser.GTE, 0)

    def NEQ(self):
        return self.getToken(LaTeXParser.NEQ, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_relation