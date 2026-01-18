from antlr4 import *
def visitPartitionVal(self, ctx: fugue_sqlParser.PartitionValContext):
    return self.visitChildren(ctx)