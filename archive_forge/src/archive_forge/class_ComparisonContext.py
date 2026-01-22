from antlr4 import *
from io import StringIO
import sys
class ComparisonContext(ValueExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.left = None
        self.right = None
        self.copyFrom(ctx)

    def comparisonOperator(self):
        return self.getTypedRuleContext(fugue_sqlParser.ComparisonOperatorContext, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComparison'):
            return visitor.visitComparison(self)
        else:
            return visitor.visitChildren(self)