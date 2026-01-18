from antlr4 import *
def visitFuguePaths(self, ctx: fugue_sqlParser.FuguePathsContext):
    return self.visitChildren(ctx)