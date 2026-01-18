from antlr4 import *
def visitFugueZipTask(self, ctx: fugue_sqlParser.FugueZipTaskContext):
    return self.visitChildren(ctx)