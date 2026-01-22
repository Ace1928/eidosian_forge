from antlr4 import *
from io import StringIO
import sys
class ShowPartitionsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def PARTITIONS(self):
        return self.getToken(fugue_sqlParser.PARTITIONS, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowPartitions'):
            return visitor.visitShowPartitions(self)
        else:
            return visitor.visitChildren(self)