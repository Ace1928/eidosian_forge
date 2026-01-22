from antlr4 import *
from io import StringIO
import sys
class FugueOutputTransformTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.dfs = None
        self.partition = None
        self.fugueUsing = None
        self.params = None
        self.callback = None

    def OUTTRANSFORM(self):
        return self.getToken(fugue_sqlParser.OUTTRANSFORM, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def fugueExtension(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueExtensionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, i)

    def CALLBACK(self):
        return self.getToken(fugue_sqlParser.CALLBACK, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueOutputTransformTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueOutputTransformTask'):
            return visitor.visitFugueOutputTransformTask(self)
        else:
            return visitor.visitChildren(self)