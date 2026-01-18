from antlr4 import *
def visitSingleStatement(self, ctx: fugue_sqlParser.SingleStatementContext):
    return self.visitChildren(ctx)