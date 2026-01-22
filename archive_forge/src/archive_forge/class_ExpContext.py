from antlr4 import *
from io import StringIO
import sys
class ExpContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def EXP(self):
        return self.getToken(AutolevParser.EXP, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterExp'):
            listener.enterExp(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitExp'):
            listener.exitExp(self)