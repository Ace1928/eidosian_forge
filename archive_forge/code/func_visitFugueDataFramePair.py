from antlr4 import *
def visitFugueDataFramePair(self, ctx: fugue_sqlParser.FugueDataFramePairContext):
    return self.visitChildren(ctx)