from antlr4 import *
def visitFuguePartitionAlgo(self, ctx: fugue_sqlParser.FuguePartitionAlgoContext):
    return self.visitChildren(ctx)