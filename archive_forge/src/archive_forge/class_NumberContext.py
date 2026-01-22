from antlr4 import *
from io import StringIO
import sys
class NumberContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DIGIT(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.DIGIT)
        else:
            return self.getToken(LaTeXParser.DIGIT, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_number