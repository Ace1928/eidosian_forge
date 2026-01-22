from antlr4 import *
from io import StringIO
import sys
class FugueSampleTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.method = None
        self.seed = None
        self.df = None
        self.by = None

    def SAMPLE(self):
        return self.getToken(fugue_sqlParser.SAMPLE, 0)

    def fugueSampleMethod(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSampleMethodContext, 0)

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def SEED(self):
        return self.getToken(fugue_sqlParser.SEED, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def PREPARTITION(self):
        return self.getToken(fugue_sqlParser.PREPARTITION, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSampleTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSampleTask'):
            return visitor.visitFugueSampleTask(self)
        else:
            return visitor.visitChildren(self)