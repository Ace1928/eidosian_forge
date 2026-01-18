from antlr4 import *
def visitQueryPrimaryDefault(self, ctx: fugue_sqlParser.QueryPrimaryDefaultContext):
    return self.visitChildren(ctx)