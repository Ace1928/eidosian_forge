from antlr4 import *
def visitComplexDataType(self, ctx: fugue_sqlParser.ComplexDataTypeContext):
    return self.visitChildren(ctx)