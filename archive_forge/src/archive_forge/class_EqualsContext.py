from antlr4 import *
from io import StringIO
import sys
class EqualsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def getRuleIndex(self):
        return AutolevParser.RULE_equals

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterEquals'):
            listener.enterEquals(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitEquals'):
            listener.exitEquals(self)