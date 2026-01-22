from antlr4 import *
from io import StringIO
import sys
class FugueModuleTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.assign = None
        self.dfs = None
        self.fugueUsing = None
        self.params = None

    def SUB(self):
        return self.getToken(fugue_sqlParser.SUB, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def fugueExtension(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

    def fugueAssignment(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentContext, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueModuleTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueModuleTask'):
            return visitor.visitFugueModuleTask(self)
        else:
            return visitor.visitChildren(self)