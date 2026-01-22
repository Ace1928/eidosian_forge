from antlr4 import *
from io import StringIO
import sys
class SubscriptContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.value = None
        self.index = None
        self.copyFrom(ctx)

    def primaryExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

    def valueExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSubscript'):
            return visitor.visitSubscript(self)
        else:
            return visitor.visitChildren(self)