from antlr4 import *
def visitFugueSingleTask(self, ctx: fugue_sqlParser.FugueSingleTaskContext):
    return self.visitChildren(ctx)