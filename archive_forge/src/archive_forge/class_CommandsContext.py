from antlr4 import *
from io import StringIO
import sys
class CommandsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Save(self):
        return self.getToken(AutolevParser.Save, 0)

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def Encode(self):
        return self.getToken(AutolevParser.Encode, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_commands

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterCommands'):
            listener.enterCommands(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitCommands'):
            listener.exitCommands(self)