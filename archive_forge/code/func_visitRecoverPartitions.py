from antlr4 import *
def visitRecoverPartitions(self, ctx: fugue_sqlParser.RecoverPartitionsContext):
    return self.visitChildren(ctx)