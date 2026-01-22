from antlr4 import *
from io import StringIO
import sys
class FuncContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.root = None
        self.base = None

    def func_normal(self):
        return self.getTypedRuleContext(LaTeXParser.Func_normalContext, 0)

    def L_PAREN(self):
        return self.getToken(LaTeXParser.L_PAREN, 0)

    def func_arg(self):
        return self.getTypedRuleContext(LaTeXParser.Func_argContext, 0)

    def R_PAREN(self):
        return self.getToken(LaTeXParser.R_PAREN, 0)

    def func_arg_noparens(self):
        return self.getTypedRuleContext(LaTeXParser.Func_arg_noparensContext, 0)

    def subexpr(self):
        return self.getTypedRuleContext(LaTeXParser.SubexprContext, 0)

    def supexpr(self):
        return self.getTypedRuleContext(LaTeXParser.SupexprContext, 0)

    def args(self):
        return self.getTypedRuleContext(LaTeXParser.ArgsContext, 0)

    def LETTER(self):
        return self.getToken(LaTeXParser.LETTER, 0)

    def SYMBOL(self):
        return self.getToken(LaTeXParser.SYMBOL, 0)

    def SINGLE_QUOTES(self):
        return self.getToken(LaTeXParser.SINGLE_QUOTES, 0)

    def FUNC_INT(self):
        return self.getToken(LaTeXParser.FUNC_INT, 0)

    def DIFFERENTIAL(self):
        return self.getToken(LaTeXParser.DIFFERENTIAL, 0)

    def frac(self):
        return self.getTypedRuleContext(LaTeXParser.FracContext, 0)

    def additive(self):
        return self.getTypedRuleContext(LaTeXParser.AdditiveContext, 0)

    def FUNC_SQRT(self):
        return self.getToken(LaTeXParser.FUNC_SQRT, 0)

    def L_BRACE(self):
        return self.getToken(LaTeXParser.L_BRACE, 0)

    def R_BRACE(self):
        return self.getToken(LaTeXParser.R_BRACE, 0)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.ExprContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.ExprContext, i)

    def L_BRACKET(self):
        return self.getToken(LaTeXParser.L_BRACKET, 0)

    def R_BRACKET(self):
        return self.getToken(LaTeXParser.R_BRACKET, 0)

    def FUNC_OVERLINE(self):
        return self.getToken(LaTeXParser.FUNC_OVERLINE, 0)

    def mp(self):
        return self.getTypedRuleContext(LaTeXParser.MpContext, 0)

    def FUNC_SUM(self):
        return self.getToken(LaTeXParser.FUNC_SUM, 0)

    def FUNC_PROD(self):
        return self.getToken(LaTeXParser.FUNC_PROD, 0)

    def subeq(self):
        return self.getTypedRuleContext(LaTeXParser.SubeqContext, 0)

    def FUNC_LIM(self):
        return self.getToken(LaTeXParser.FUNC_LIM, 0)

    def limit_sub(self):
        return self.getTypedRuleContext(LaTeXParser.Limit_subContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_func