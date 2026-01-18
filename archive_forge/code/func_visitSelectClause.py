from antlr4 import *
def visitSelectClause(self, ctx: fugue_sqlParser.SelectClauseContext):
    return self.visitChildren(ctx)