from antlr4 import *
def visitPrimitiveDataType(self, ctx: fugue_sqlParser.PrimitiveDataTypeContext):
    return self.visitChildren(ctx)