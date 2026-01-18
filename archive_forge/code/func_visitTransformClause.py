from antlr4 import *
def visitTransformClause(self, ctx: fugue_sqlParser.TransformClauseContext):
    return self.visitChildren(ctx)