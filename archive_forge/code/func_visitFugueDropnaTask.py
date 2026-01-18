from antlr4 import *
def visitFugueDropnaTask(self, ctx: fugue_sqlParser.FugueDropnaTaskContext):
    return self.visitChildren(ctx)