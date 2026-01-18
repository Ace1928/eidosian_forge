from antlr4 import *
def visitOverlay(self, ctx: fugue_sqlParser.OverlayContext):
    return self.visitChildren(ctx)