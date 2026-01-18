from antlr4 import *
def visitFugueOutputTask(self, ctx: fugue_sqlParser.FugueOutputTaskContext):
    return self.visitChildren(ctx)