from antlr4 import *
from io import StringIO
import sys
class WhereClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def WHERE(self):
        return self.getToken(fugue_sqlParser.WHERE, 0)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_whereClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWhereClause'):
            return visitor.visitWhereClause(self)
        else:
            return visitor.visitChildren(self)