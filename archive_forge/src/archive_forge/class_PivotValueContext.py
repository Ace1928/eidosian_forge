from antlr4 import *
from io import StringIO
import sys
class PivotValueContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_pivotValue

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPivotValue'):
            return visitor.visitPivotValue(self)
        else:
            return visitor.visitChildren(self)