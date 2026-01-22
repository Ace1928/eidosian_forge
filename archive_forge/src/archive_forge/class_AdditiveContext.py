from antlr4 import *
from io import StringIO
import sys
class AdditiveContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def mp(self):
        return self.getTypedRuleContext(LaTeXParser.MpContext, 0)

    def additive(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.AdditiveContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.AdditiveContext, i)

    def ADD(self):
        return self.getToken(LaTeXParser.ADD, 0)

    def SUB(self):
        return self.getToken(LaTeXParser.SUB, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_additive