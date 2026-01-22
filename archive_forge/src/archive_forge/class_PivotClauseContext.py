from antlr4 import *
from io import StringIO
import sys
class PivotClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.aggregates = None
        self._pivotValue = None
        self.pivotValues = list()

    def PIVOT(self):
        return self.getToken(fugue_sqlParser.PIVOT, 0)

    def FOR(self):
        return self.getToken(fugue_sqlParser.FOR, 0)

    def pivotColumn(self):
        return self.getTypedRuleContext(fugue_sqlParser.PivotColumnContext, 0)

    def IN(self):
        return self.getToken(fugue_sqlParser.IN, 0)

    def namedExpressionSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

    def pivotValue(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PivotValueContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PivotValueContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_pivotClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPivotClause'):
            return visitor.visitPivotClause(self)
        else:
            return visitor.visitChildren(self)