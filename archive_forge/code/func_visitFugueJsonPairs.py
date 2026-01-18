from antlr4 import *
def visitFugueJsonPairs(self, ctx: fugue_sqlParser.FugueJsonPairsContext):
    return self.visitChildren(ctx)