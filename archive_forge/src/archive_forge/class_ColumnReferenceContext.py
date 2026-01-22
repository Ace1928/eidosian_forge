from antlr4 import *
from io import StringIO
import sys
class ColumnReferenceContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitColumnReference'):
            return visitor.visitColumnReference(self)
        else:
            return visitor.visitChildren(self)