from antlr4 import *
def visitPartitionSpec(self, ctx: fugue_sqlParser.PartitionSpecContext):
    return self.visitChildren(ctx)