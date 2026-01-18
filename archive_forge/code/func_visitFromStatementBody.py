from antlr4 import *
def visitFromStatementBody(self, ctx: fugue_sqlParser.FromStatementBodyContext):
    return self.visitChildren(ctx)