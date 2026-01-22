from antlr4 import *
from io import StringIO
import sys
class SampleByRowsContext(SampleMethodContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSampleByRows'):
            return visitor.visitSampleByRows(self)
        else:
            return visitor.visitChildren(self)