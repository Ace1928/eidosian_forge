from antlr4 import *
from io import StringIO
import sys
class Func_normalContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def FUNC_EXP(self):
        return self.getToken(LaTeXParser.FUNC_EXP, 0)

    def FUNC_LOG(self):
        return self.getToken(LaTeXParser.FUNC_LOG, 0)

    def FUNC_LG(self):
        return self.getToken(LaTeXParser.FUNC_LG, 0)

    def FUNC_LN(self):
        return self.getToken(LaTeXParser.FUNC_LN, 0)

    def FUNC_SIN(self):
        return self.getToken(LaTeXParser.FUNC_SIN, 0)

    def FUNC_COS(self):
        return self.getToken(LaTeXParser.FUNC_COS, 0)

    def FUNC_TAN(self):
        return self.getToken(LaTeXParser.FUNC_TAN, 0)

    def FUNC_CSC(self):
        return self.getToken(LaTeXParser.FUNC_CSC, 0)

    def FUNC_SEC(self):
        return self.getToken(LaTeXParser.FUNC_SEC, 0)

    def FUNC_COT(self):
        return self.getToken(LaTeXParser.FUNC_COT, 0)

    def FUNC_ARCSIN(self):
        return self.getToken(LaTeXParser.FUNC_ARCSIN, 0)

    def FUNC_ARCCOS(self):
        return self.getToken(LaTeXParser.FUNC_ARCCOS, 0)

    def FUNC_ARCTAN(self):
        return self.getToken(LaTeXParser.FUNC_ARCTAN, 0)

    def FUNC_ARCCSC(self):
        return self.getToken(LaTeXParser.FUNC_ARCCSC, 0)

    def FUNC_ARCSEC(self):
        return self.getToken(LaTeXParser.FUNC_ARCSEC, 0)

    def FUNC_ARCCOT(self):
        return self.getToken(LaTeXParser.FUNC_ARCCOT, 0)

    def FUNC_SINH(self):
        return self.getToken(LaTeXParser.FUNC_SINH, 0)

    def FUNC_COSH(self):
        return self.getToken(LaTeXParser.FUNC_COSH, 0)

    def FUNC_TANH(self):
        return self.getToken(LaTeXParser.FUNC_TANH, 0)

    def FUNC_ARSINH(self):
        return self.getToken(LaTeXParser.FUNC_ARSINH, 0)

    def FUNC_ARCOSH(self):
        return self.getToken(LaTeXParser.FUNC_ARCOSH, 0)

    def FUNC_ARTANH(self):
        return self.getToken(LaTeXParser.FUNC_ARTANH, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_func_normal