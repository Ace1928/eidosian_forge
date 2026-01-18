from antlr4 import *
def visitWhereClause(self, ctx: fugue_sqlParser.WhereClauseContext):
    return self.visitChildren(ctx)