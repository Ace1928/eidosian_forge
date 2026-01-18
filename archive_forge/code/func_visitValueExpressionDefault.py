from antlr4 import *
def visitValueExpressionDefault(self, ctx: fugue_sqlParser.ValueExpressionDefaultContext):
    return self.visitChildren(ctx)