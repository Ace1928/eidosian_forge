from antlr4 import *
from io import StringIO
import sys
class DropTablePartitionsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def partitionSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, i)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def PURGE(self):
        return self.getToken(fugue_sqlParser.PURGE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDropTablePartitions'):
            return visitor.visitDropTablePartitions(self)
        else:
            return visitor.visitChildren(self)