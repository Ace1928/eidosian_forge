from antlr4 import *
from io import StringIO
import sys
class InputsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Input(self):
        return self.getToken(AutolevParser.Input, 0)

    def inputs2(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.Inputs2Context)
        else:
            return self.getTypedRuleContext(AutolevParser.Inputs2Context, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_inputs

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterInputs'):
            listener.enterInputs(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitInputs'):
            listener.exitInputs(self)