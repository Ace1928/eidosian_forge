from antlr4 import *
from io import StringIO
import sys
class DereferenceContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.base = None
        self.fieldName = None
        self.copyFrom(ctx)

    def primaryExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDereference'):
            return visitor.visitDereference(self)
        else:
            return visitor.visitChildren(self)