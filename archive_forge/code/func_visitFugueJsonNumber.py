from antlr4 import *
def visitFugueJsonNumber(self, ctx: fugue_sqlParser.FugueJsonNumberContext):
    return self.visitChildren(ctx)