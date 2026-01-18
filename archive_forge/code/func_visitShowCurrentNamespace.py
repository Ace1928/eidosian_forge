from antlr4 import *
def visitShowCurrentNamespace(self, ctx: fugue_sqlParser.ShowCurrentNamespaceContext):
    return self.visitChildren(ctx)