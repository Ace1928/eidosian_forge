from antlr4 import *
from io import StringIO
import sys
class FugueNestableTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.assign = None
        self.q = None
        self.checkpoint = None
        self.broadcast = None
        self.y = None

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def fugueAssignment(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentContext, 0)

    def fugueCheckpoint(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueCheckpointContext, 0)

    def fugueBroadcast(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueBroadcastContext, 0)

    def fugueYield(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueYieldContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueNestableTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueNestableTask'):
            return visitor.visitFugueNestableTask(self)
        else:
            return visitor.visitChildren(self)