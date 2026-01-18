from antlr4 import *
def visitRelation(self, ctx: fugue_sqlParser.RelationContext):
    return self.visitChildren(ctx)