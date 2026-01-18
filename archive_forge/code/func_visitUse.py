from antlr4 import *
def visitUse(self, ctx: fugue_sqlParser.UseContext):
    return self.visitChildren(ctx)