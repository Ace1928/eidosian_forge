from antlr4 import *
def visitFugueCheckpointNamespace(self, ctx: fugue_sqlParser.FugueCheckpointNamespaceContext):
    return self.visitChildren(ctx)