from antlr4 import *
def visitJoinRelation(self, ctx: fugue_sqlParser.JoinRelationContext):
    return self.visitChildren(ctx)