from antlr4 import *
from io import StringIO
import sys
class PredicateContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.kind = None
        self.lower = None
        self.upper = None
        self.pattern = None
        self.quantifier = None
        self.escapeChar = None
        self.right = None

    def AND(self):
        return self.getToken(fugue_sqlParser.AND, 0)

    def BETWEEN(self):
        return self.getToken(fugue_sqlParser.BETWEEN, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def IN(self):
        return self.getToken(fugue_sqlParser.IN, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def RLIKE(self):
        return self.getToken(fugue_sqlParser.RLIKE, 0)

    def LIKE(self):
        return self.getToken(fugue_sqlParser.LIKE, 0)

    def ANY(self):
        return self.getToken(fugue_sqlParser.ANY, 0)

    def SOME(self):
        return self.getToken(fugue_sqlParser.SOME, 0)

    def ALL(self):
        return self.getToken(fugue_sqlParser.ALL, 0)

    def ESCAPE(self):
        return self.getToken(fugue_sqlParser.ESCAPE, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def IS(self):
        return self.getToken(fugue_sqlParser.IS, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def TRUE(self):
        return self.getToken(fugue_sqlParser.TRUE, 0)

    def FALSE(self):
        return self.getToken(fugue_sqlParser.FALSE, 0)

    def UNKNOWN(self):
        return self.getToken(fugue_sqlParser.UNKNOWN, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def DISTINCT(self):
        return self.getToken(fugue_sqlParser.DISTINCT, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_predicate

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPredicate'):
            return visitor.visitPredicate(self)
        else:
            return visitor.visitChildren(self)