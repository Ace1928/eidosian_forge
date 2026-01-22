from antlr4 import *
from io import StringIO
import sys
class FuguePartitionNumberContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def ROWCOUNT(self):
        return self.getToken(fugue_sqlParser.ROWCOUNT, 0)

    def CONCURRENCY(self):
        return self.getToken(fugue_sqlParser.CONCURRENCY, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fuguePartitionNumber

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFuguePartitionNumber'):
            return visitor.visitFuguePartitionNumber(self)
        else:
            return visitor.visitChildren(self)