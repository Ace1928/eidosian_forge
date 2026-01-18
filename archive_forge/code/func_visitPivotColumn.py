from antlr4 import *
def visitPivotColumn(self, ctx: fugue_sqlParser.PivotColumnContext):
    return self.visitChildren(ctx)