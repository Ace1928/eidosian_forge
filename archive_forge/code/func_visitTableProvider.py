from antlr4 import *
def visitTableProvider(self, ctx: fugue_sqlParser.TableProviderContext):
    return self.visitChildren(ctx)