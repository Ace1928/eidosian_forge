from antlr4 import *
from io import StringIO
import sys
class OverlayContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.iinput = None
        self.replace = None
        self.position = None
        self.length = None
        self.copyFrom(ctx)

    def OVERLAY(self):
        return self.getToken(fugue_sqlParser.OVERLAY, 0)

    def PLACING(self):
        return self.getToken(fugue_sqlParser.PLACING, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def FOR(self):
        return self.getToken(fugue_sqlParser.FOR, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitOverlay'):
            return visitor.visitOverlay(self)
        else:
            return visitor.visitChildren(self)