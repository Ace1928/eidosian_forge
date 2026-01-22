from antlr4 import *
from io import StringIO
import sys
class FugueTakeTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.rows = None
        self.df = None
        self.partition = None
        self.presort = None
        self.na_position = None

    def TAKE(self):
        return self.getToken(fugue_sqlParser.TAKE, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def PRESORT(self):
        return self.getToken(fugue_sqlParser.PRESORT, 0)

    def FIRST(self):
        return self.getToken(fugue_sqlParser.FIRST, 0)

    def LAST(self):
        return self.getToken(fugue_sqlParser.LAST, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def fugueColsSort(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueTakeTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueTakeTask'):
            return visitor.visitFugueTakeTask(self)
        else:
            return visitor.visitChildren(self)