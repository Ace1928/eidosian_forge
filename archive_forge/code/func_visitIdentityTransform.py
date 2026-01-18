from antlr4 import *
def visitIdentityTransform(self, ctx: fugue_sqlParser.IdentityTransformContext):
    return self.visitChildren(ctx)