from antlr4 import *
from io import StringIO
import sys
class IntContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def INT(self):
        return self.getToken(AutolevParser.INT, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterInt'):
            listener.enterInt(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitInt'):
            listener.exitInt(self)