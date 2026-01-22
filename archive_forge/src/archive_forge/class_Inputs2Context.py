from antlr4 import *
from io import StringIO
import sys
class Inputs2Context(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def id_diff(self):
        return self.getTypedRuleContext(AutolevParser.Id_diffContext, 0)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.ExprContext)
        else:
            return self.getTypedRuleContext(AutolevParser.ExprContext, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_inputs2

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterInputs2'):
            listener.enterInputs2(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitInputs2'):
            listener.exitInputs2(self)