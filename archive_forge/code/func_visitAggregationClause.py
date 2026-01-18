from antlr4 import *
def visitAggregationClause(self, ctx: fugue_sqlParser.AggregationClauseContext):
    return self.visitChildren(ctx)