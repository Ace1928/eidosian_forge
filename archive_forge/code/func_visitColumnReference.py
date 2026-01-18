from antlr4 import *
def visitColumnReference(self, ctx: fugue_sqlParser.ColumnReferenceContext):
    return self.visitChildren(ctx)