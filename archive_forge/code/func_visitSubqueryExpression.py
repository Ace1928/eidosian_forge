from antlr4 import *
def visitSubqueryExpression(self, ctx: fugue_sqlParser.SubqueryExpressionContext):
    return self.visitChildren(ctx)