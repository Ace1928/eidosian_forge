from antlr4 import *
def visitConstantDefault(self, ctx: fugue_sqlParser.ConstantDefaultContext):
    return self.visitChildren(ctx)