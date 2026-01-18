from antlr4 import *
def visitRowFormatDelimited(self, ctx: fugue_sqlParser.RowFormatDelimitedContext):
    return self.visitChildren(ctx)