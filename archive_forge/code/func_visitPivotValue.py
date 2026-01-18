from antlr4 import *
def visitPivotValue(self, ctx: fugue_sqlParser.PivotValueContext):
    return self.visitChildren(ctx)