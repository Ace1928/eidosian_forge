from antlr4 import *
def visitComparisonOperator(self, ctx: fugue_sqlParser.ComparisonOperatorContext):
    return self.visitChildren(ctx)