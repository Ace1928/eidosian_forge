from antlr4 import *
from io import StringIO
import sys
class FuguePrepartitionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.algo = None
        self.num = None
        self.by = None
        self.presort = None

    def PREPARTITION(self):
        return self.getToken(fugue_sqlParser.PREPARTITION, 0)

    def fuguePartitionNum(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumContext, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def PRESORT(self):
        return self.getToken(fugue_sqlParser.PRESORT, 0)

    def fuguePartitionAlgo(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionAlgoContext, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def fugueColsSort(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fuguePrepartition

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFuguePrepartition'):
            return visitor.visitFuguePrepartition(self)
        else:
            return visitor.visitChildren(self)