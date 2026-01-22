from antlr4 import *
from io import StringIO
import sys
class OutputsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Output(self):
        return self.getToken(AutolevParser.Output, 0)

    def outputs2(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.Outputs2Context)
        else:
            return self.getTypedRuleContext(AutolevParser.Outputs2Context, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_outputs

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterOutputs'):
            listener.enterOutputs(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitOutputs'):
            listener.exitOutputs(self)