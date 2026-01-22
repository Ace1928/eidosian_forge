from antlr4 import *
from io import StringIO
import sys
class FirstContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def FIRST(self):
        return self.getToken(fugue_sqlParser.FIRST, 0)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def IGNORE(self):
        return self.getToken(fugue_sqlParser.IGNORE, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFirst'):
            return visitor.visitFirst(self)
        else:
            return visitor.visitChildren(self)