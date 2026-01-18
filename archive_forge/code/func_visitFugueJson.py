from antlr4 import *
def visitFugueJson(self, ctx: fugue_sqlParser.FugueJsonContext):
    return self.visitChildren(ctx)