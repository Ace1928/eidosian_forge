from antlr4 import *
from io import StringIO
import sys
class AtomContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def LETTER(self):
        return self.getToken(LaTeXParser.LETTER, 0)

    def SYMBOL(self):
        return self.getToken(LaTeXParser.SYMBOL, 0)

    def subexpr(self):
        return self.getTypedRuleContext(LaTeXParser.SubexprContext, 0)

    def SINGLE_QUOTES(self):
        return self.getToken(LaTeXParser.SINGLE_QUOTES, 0)

    def number(self):
        return self.getTypedRuleContext(LaTeXParser.NumberContext, 0)

    def DIFFERENTIAL(self):
        return self.getToken(LaTeXParser.DIFFERENTIAL, 0)

    def mathit(self):
        return self.getTypedRuleContext(LaTeXParser.MathitContext, 0)

    def frac(self):
        return self.getTypedRuleContext(LaTeXParser.FracContext, 0)

    def binom(self):
        return self.getTypedRuleContext(LaTeXParser.BinomContext, 0)

    def bra(self):
        return self.getTypedRuleContext(LaTeXParser.BraContext, 0)

    def ket(self):
        return self.getTypedRuleContext(LaTeXParser.KetContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_atom