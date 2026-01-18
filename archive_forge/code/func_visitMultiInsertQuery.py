from antlr4 import *
def visitMultiInsertQuery(self, ctx: fugue_sqlParser.MultiInsertQueryContext):
    return self.visitChildren(ctx)