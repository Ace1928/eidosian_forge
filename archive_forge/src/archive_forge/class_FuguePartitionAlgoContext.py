from antlr4 import *
from io import StringIO
import sys
class FuguePartitionAlgoContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def HASH(self):
        return self.getToken(fugue_sqlParser.HASH, 0)

    def RAND(self):
        return self.getToken(fugue_sqlParser.RAND, 0)

    def EVEN(self):
        return self.getToken(fugue_sqlParser.EVEN, 0)

    def COARSE(self):
        return self.getToken(fugue_sqlParser.COARSE, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fuguePartitionAlgo

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFuguePartitionAlgo'):
            return visitor.visitFuguePartitionAlgo(self)
        else:
            return visitor.visitChildren(self)