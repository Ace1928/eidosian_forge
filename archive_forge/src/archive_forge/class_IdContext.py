from antlr4 import *
from io import StringIO
import sys
class IdContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterId'):
            listener.enterId(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitId'):
            listener.exitId(self)