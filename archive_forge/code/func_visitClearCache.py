from antlr4 import *
def visitClearCache(self, ctx: fugue_sqlParser.ClearCacheContext):
    return self.visitChildren(ctx)