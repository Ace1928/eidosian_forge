from antlr4 import *
def visitComplexColTypeList(self, ctx: fugue_sqlParser.ComplexColTypeListContext):
    return self.visitChildren(ctx)