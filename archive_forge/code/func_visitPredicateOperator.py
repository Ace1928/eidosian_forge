from antlr4 import *
def visitPredicateOperator(self, ctx: fugue_sqlParser.PredicateOperatorContext):
    return self.visitChildren(ctx)