from antlr4 import *
from io import StringIO
import sys
class FugueCheckpointDeterministicContext(FugueCheckpointContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.ns = None
        self.partition = None
        self.single = None
        self.params = None
        self.copyFrom(ctx)

    def DETERMINISTIC(self):
        return self.getToken(fugue_sqlParser.DETERMINISTIC, 0)

    def CHECKPOINT(self):
        return self.getToken(fugue_sqlParser.CHECKPOINT, 0)

    def LAZY(self):
        return self.getToken(fugue_sqlParser.LAZY, 0)

    def fugueCheckpointNamespace(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueCheckpointNamespaceContext, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def fugueSingleFile(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueCheckpointDeterministic'):
            return visitor.visitFugueCheckpointDeterministic(self)
        else:
            return visitor.visitChildren(self)