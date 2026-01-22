from antlr4 import *
from io import StringIO
import sys
class ComparisonOperatorContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def comparisonEqualOperator(self):
        return self.getTypedRuleContext(fugue_sqlParser.ComparisonEqualOperatorContext, 0)

    def NEQ(self):
        return self.getToken(fugue_sqlParser.NEQ, 0)

    def NEQJ(self):
        return self.getToken(fugue_sqlParser.NEQJ, 0)

    def LT(self):
        return self.getToken(fugue_sqlParser.LT, 0)

    def LTE(self):
        return self.getToken(fugue_sqlParser.LTE, 0)

    def GT(self):
        return self.getToken(fugue_sqlParser.GT, 0)

    def GTE(self):
        return self.getToken(fugue_sqlParser.GTE, 0)

    def NSEQ(self):
        return self.getToken(fugue_sqlParser.NSEQ, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_comparisonOperator

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComparisonOperator'):
            return visitor.visitComparisonOperator(self)
        else:
            return visitor.visitChildren(self)