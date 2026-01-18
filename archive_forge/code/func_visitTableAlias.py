from antlr4 import *
def visitTableAlias(self, ctx: fugue_sqlParser.TableAliasContext):
    return self.visitChildren(ctx)