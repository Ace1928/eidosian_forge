from antlr4 import *
from io import StringIO
import sys
class StatContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def varDecl(self):
        return self.getTypedRuleContext(AutolevParser.VarDeclContext, 0)

    def functionCall(self):
        return self.getTypedRuleContext(AutolevParser.FunctionCallContext, 0)

    def codeCommands(self):
        return self.getTypedRuleContext(AutolevParser.CodeCommandsContext, 0)

    def massDecl(self):
        return self.getTypedRuleContext(AutolevParser.MassDeclContext, 0)

    def inertiaDecl(self):
        return self.getTypedRuleContext(AutolevParser.InertiaDeclContext, 0)

    def assignment(self):
        return self.getTypedRuleContext(AutolevParser.AssignmentContext, 0)

    def settings(self):
        return self.getTypedRuleContext(AutolevParser.SettingsContext, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_stat

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterStat'):
            listener.enterStat(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitStat'):
            listener.exitStat(self)