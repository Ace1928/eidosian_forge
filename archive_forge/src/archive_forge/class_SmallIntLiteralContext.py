from antlr4 import *
from io import StringIO
import sys
class SmallIntLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def SMALLINT_LITERAL(self):
        return self.getToken(fugue_sqlParser.SMALLINT_LITERAL, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSmallIntLiteral'):
            return visitor.visitSmallIntLiteral(self)
        else:
            return visitor.visitChildren(self)