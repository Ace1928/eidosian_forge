from antlr4 import *
def visitTransformArgument(self, ctx: fugue_sqlParser.TransformArgumentContext):
    return self.visitChildren(ctx)