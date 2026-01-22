from antlr4 import *
from io import StringIO
import sys
class PostfixContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def exp(self):
        return self.getTypedRuleContext(LaTeXParser.ExpContext, 0)

    def postfix_op(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.Postfix_opContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.Postfix_opContext, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_postfix