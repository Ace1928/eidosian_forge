from antlr4 import *
def visitConstantList(self, ctx: fugue_sqlParser.ConstantListContext):
    return self.visitChildren(ctx)