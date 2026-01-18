from antlr4 import *
def visitNestedConstantList(self, ctx: fugue_sqlParser.NestedConstantListContext):
    return self.visitChildren(ctx)