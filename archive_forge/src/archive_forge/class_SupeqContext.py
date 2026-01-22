from antlr4 import *
from io import StringIO
import sys
class SupeqContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def UNDERSCORE(self):
        return self.getToken(LaTeXParser.UNDERSCORE, 0)

    def L_BRACE(self):
        return self.getToken(LaTeXParser.L_BRACE, 0)

    def equality(self):
        return self.getTypedRuleContext(LaTeXParser.EqualityContext, 0)

    def R_BRACE(self):
        return self.getToken(LaTeXParser.R_BRACE, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_supeq