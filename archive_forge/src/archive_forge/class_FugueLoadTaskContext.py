from antlr4 import *
from io import StringIO
import sys
class FugueLoadTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.fmt = None
        self.path = None
        self.paths = None
        self.params = None
        self.columns = None

    def LOAD(self):
        return self.getToken(fugue_sqlParser.LOAD, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def fugueFileFormat(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def fugueLoadColumns(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueLoadColumnsContext, 0)

    def fuguePath(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, 0)

    def fuguePaths(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePathsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueLoadTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueLoadTask'):
            return visitor.visitFugueLoadTask(self)
        else:
            return visitor.visitChildren(self)