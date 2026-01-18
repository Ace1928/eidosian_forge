from antlr4 import *
def visitAliasedQuery(self, ctx: fugue_sqlParser.AliasedQueryContext):
    return self.visitChildren(ctx)