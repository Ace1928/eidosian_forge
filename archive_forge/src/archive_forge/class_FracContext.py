from antlr4 import *
from io import StringIO
import sys
class FracContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.upperd = None
        self.upper = None
        self.lowerd = None
        self.lower = None

    def CMD_FRAC(self):
        return self.getToken(LaTeXParser.CMD_FRAC, 0)

    def L_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.L_BRACE)
        else:
            return self.getToken(LaTeXParser.L_BRACE, i)

    def R_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.R_BRACE)
        else:
            return self.getToken(LaTeXParser.R_BRACE, i)

    def DIGIT(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.DIGIT)
        else:
            return self.getToken(LaTeXParser.DIGIT, i)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.ExprContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.ExprContext, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_frac