from antlr4 import *
def visitSetClause(self, ctx: fugue_sqlParser.SetClauseContext):
    return self.visitChildren(ctx)