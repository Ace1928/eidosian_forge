from antlr4 import *
def visitFugueCheckpointWeak(self, ctx: fugue_sqlParser.FugueCheckpointWeakContext):
    return self.visitChildren(ctx)