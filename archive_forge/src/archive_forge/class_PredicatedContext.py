from antlr4 import *
from io import StringIO
import sys
class PredicatedContext(BooleanExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def valueExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

    def predicate(self):
        return self.getTypedRuleContext(fugue_sqlParser.PredicateContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPredicated'):
            return visitor.visitPredicated(self)
        else:
            return visitor.visitChildren(self)