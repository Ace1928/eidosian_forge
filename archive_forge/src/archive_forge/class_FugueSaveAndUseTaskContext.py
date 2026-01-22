from antlr4 import *
from io import StringIO
import sys
class FugueSaveAndUseTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.df = None
        self.partition = None
        self.m = None
        self.single = None
        self.fmt = None
        self.path = None
        self.params = None

    def SAVE(self):
        return self.getToken(fugue_sqlParser.SAVE, 0)

    def AND(self):
        return self.getToken(fugue_sqlParser.AND, 0)

    def USE(self):
        return self.getToken(fugue_sqlParser.USE, 0)

    def fugueSaveMode(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSaveModeContext, 0)

    def fuguePath(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def fugueSingleFile(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

    def fugueFileFormat(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSaveAndUseTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSaveAndUseTask'):
            return visitor.visitFugueSaveAndUseTask(self)
        else:
            return visitor.visitChildren(self)