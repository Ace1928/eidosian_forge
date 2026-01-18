from antlr4 import *
def visitFugueBroadcast(self, ctx: fugue_sqlParser.FugueBroadcastContext):
    return self.visitChildren(ctx)