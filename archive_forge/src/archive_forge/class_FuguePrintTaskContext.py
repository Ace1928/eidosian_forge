from antlr4 import *
from io import StringIO
import sys
class FuguePrintTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.rows = None
        self.dfs = None
        self.count = None
        self.title = None

    def PRINT(self):
        return self.getToken(fugue_sqlParser.PRINT, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def TITLE(self):
        return self.getToken(fugue_sqlParser.TITLE, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def ROWCOUNT(self):
        return self.getToken(fugue_sqlParser.ROWCOUNT, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fuguePrintTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFuguePrintTask'):
            return visitor.visitFuguePrintTask(self)
        else:
            return visitor.visitChildren(self)