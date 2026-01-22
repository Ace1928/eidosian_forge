from antlr4 import *
from io import StringIO
import sys
class NullLiteralContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNullLiteral'):
            return visitor.visitNullLiteral(self)
        else:
            return visitor.visitChildren(self)