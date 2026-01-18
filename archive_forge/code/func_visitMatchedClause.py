from antlr4 import *
def visitMatchedClause(self, ctx: fugue_sqlParser.MatchedClauseContext):
    return self.visitChildren(ctx)