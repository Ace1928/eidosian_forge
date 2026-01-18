from antlr4 import *
def visitSingleDataType(self, ctx: fugue_sqlParser.SingleDataTypeContext):
    return self.visitChildren(ctx)