from antlr4 import *
def visitFugueTakeTask(self, ctx: fugue_sqlParser.FugueTakeTaskContext):
    return self.visitChildren(ctx)