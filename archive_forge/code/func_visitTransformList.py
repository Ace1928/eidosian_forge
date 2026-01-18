from antlr4 import *
def visitTransformList(self, ctx: fugue_sqlParser.TransformListContext):
    return self.visitChildren(ctx)