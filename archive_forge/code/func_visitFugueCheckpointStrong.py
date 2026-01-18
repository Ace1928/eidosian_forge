from antlr4 import *
def visitFugueCheckpointStrong(self, ctx: fugue_sqlParser.FugueCheckpointStrongContext):
    return self.visitChildren(ctx)