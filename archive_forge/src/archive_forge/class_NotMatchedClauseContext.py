from antlr4 import *
from io import StringIO
import sys
class NotMatchedClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.notMatchedCond = None

    def WHEN(self):
        return self.getToken(fugue_sqlParser.WHEN, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def MATCHED(self):
        return self.getToken(fugue_sqlParser.MATCHED, 0)

    def THEN(self):
        return self.getToken(fugue_sqlParser.THEN, 0)

    def notMatchedAction(self):
        return self.getTypedRuleContext(fugue_sqlParser.NotMatchedActionContext, 0)

    def AND(self):
        return self.getToken(fugue_sqlParser.AND, 0)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_notMatchedClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNotMatchedClause'):
            return visitor.visitNotMatchedClause(self)
        else:
            return visitor.visitChildren(self)