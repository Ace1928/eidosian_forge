from antlr4 import *
def visitFugueSaveAndUseTask(self, ctx: fugue_sqlParser.FugueSaveAndUseTaskContext):
    return self.visitChildren(ctx)