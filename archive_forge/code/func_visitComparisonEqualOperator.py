from antlr4 import *
def visitComparisonEqualOperator(self, ctx: fugue_sqlParser.ComparisonEqualOperatorContext):
    return self.visitChildren(ctx)