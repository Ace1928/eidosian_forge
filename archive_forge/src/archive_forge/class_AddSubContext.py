from antlr4 import *
from io import StringIO
import sys
class AddSubContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.ExprContext)
        else:
            return self.getTypedRuleContext(AutolevParser.ExprContext, i)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterAddSub'):
            listener.enterAddSub(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitAddSub'):
            listener.exitAddSub(self)