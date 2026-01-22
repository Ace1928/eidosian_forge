from antlr4 import *
from io import StringIO
import sys
class StarContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStar'):
            return visitor.visitStar(self)
        else:
            return visitor.visitChildren(self)