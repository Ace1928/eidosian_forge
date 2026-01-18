from antlr4 import *
def visitRenameTablePartition(self, ctx: fugue_sqlParser.RenameTablePartitionContext):
    return self.visitChildren(ctx)