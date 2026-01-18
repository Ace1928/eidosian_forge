from antlr4 import *
def visitColPosition(self, ctx: fugue_sqlParser.ColPositionContext):
    return self.visitChildren(ctx)