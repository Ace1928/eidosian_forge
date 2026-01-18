from antlr4 import *
def visitFugueSchemaPair(self, ctx: fugue_sqlParser.FugueSchemaPairContext):
    return self.visitChildren(ctx)