from antlr4 import *
from io import StringIO
import sys
class JoinCriteriaContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def identifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_joinCriteria

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitJoinCriteria'):
            return visitor.visitJoinCriteria(self)
        else:
            return visitor.visitChildren(self)