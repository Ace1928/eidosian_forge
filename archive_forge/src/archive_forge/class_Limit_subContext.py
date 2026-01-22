from antlr4 import *
from io import StringIO
import sys
class Limit_subContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def UNDERSCORE(self):
        return self.getToken(LaTeXParser.UNDERSCORE, 0)

    def L_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.L_BRACE)
        else:
            return self.getToken(LaTeXParser.L_BRACE, i)

    def LIM_APPROACH_SYM(self):
        return self.getToken(LaTeXParser.LIM_APPROACH_SYM, 0)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def R_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.R_BRACE)
        else:
            return self.getToken(LaTeXParser.R_BRACE, i)

    def LETTER(self):
        return self.getToken(LaTeXParser.LETTER, 0)

    def SYMBOL(self):
        return self.getToken(LaTeXParser.SYMBOL, 0)

    def CARET(self):
        return self.getToken(LaTeXParser.CARET, 0)

    def ADD(self):
        return self.getToken(LaTeXParser.ADD, 0)

    def SUB(self):
        return self.getToken(LaTeXParser.SUB, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_limit_sub