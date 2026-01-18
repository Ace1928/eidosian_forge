from antlr4 import *
def visitPosition(self, ctx: fugue_sqlParser.PositionContext):
    return self.visitChildren(ctx)