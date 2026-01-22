from antlr4 import *
from io import StringIO
import sys
class ArgsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def args(self):
        return self.getTypedRuleContext(LaTeXParser.ArgsContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_args