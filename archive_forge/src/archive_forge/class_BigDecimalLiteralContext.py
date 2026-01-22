from antlr4 import *
from io import StringIO
import sys
class BigDecimalLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def BIGDECIMAL_LITERAL(self):
        return self.getToken(fugue_sqlParser.BIGDECIMAL_LITERAL, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitBigDecimalLiteral'):
            return visitor.visitBigDecimalLiteral(self)
        else:
            return visitor.visitChildren(self)