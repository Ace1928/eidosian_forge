from antlr4 import *
from io import StringIO
import sys
class RowConstructorContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def namedExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRowConstructor'):
            return visitor.visitRowConstructor(self)
        else:
            return visitor.visitChildren(self)