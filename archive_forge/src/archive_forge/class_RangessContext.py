from antlr4 import *
from io import StringIO
import sys
class RangessContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ranges(self):
        return self.getTypedRuleContext(AutolevParser.RangesContext, 0)

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterRangess'):
            listener.enterRangess(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitRangess'):
            listener.exitRangess(self)