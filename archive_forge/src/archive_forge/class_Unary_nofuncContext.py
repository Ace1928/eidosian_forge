from antlr4 import *
from io import StringIO
import sys
class Unary_nofuncContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def unary_nofunc(self):
        return self.getTypedRuleContext(LaTeXParser.Unary_nofuncContext, 0)

    def ADD(self):
        return self.getToken(LaTeXParser.ADD, 0)

    def SUB(self):
        return self.getToken(LaTeXParser.SUB, 0)

    def postfix(self):
        return self.getTypedRuleContext(LaTeXParser.PostfixContext, 0)

    def postfix_nofunc(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.Postfix_nofuncContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.Postfix_nofuncContext, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_unary_nofunc