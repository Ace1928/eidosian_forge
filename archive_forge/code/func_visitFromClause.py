from antlr4 import *
def visitFromClause(self, ctx: fugue_sqlParser.FromClauseContext):
    return self.visitChildren(ctx)