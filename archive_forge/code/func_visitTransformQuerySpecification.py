from antlr4 import *
def visitTransformQuerySpecification(self, ctx: fugue_sqlParser.TransformQuerySpecificationContext):
    return self.visitChildren(ctx)