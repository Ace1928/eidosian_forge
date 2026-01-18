from antlr4 import *
def visitFromStatement(self, ctx: fugue_sqlParser.FromStatementContext):
    return self.visitChildren(ctx)