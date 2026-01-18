from antlr4 import *
def visitRowConstructor(self, ctx: fugue_sqlParser.RowConstructorContext):
    return self.visitChildren(ctx)