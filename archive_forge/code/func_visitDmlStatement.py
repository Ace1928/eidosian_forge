from antlr4 import *
def visitDmlStatement(self, ctx: fugue_sqlParser.DmlStatementContext):
    return self.visitChildren(ctx)