from antlr4 import *
def visitShowNamespaces(self, ctx: fugue_sqlParser.ShowNamespacesContext):
    return self.visitChildren(ctx)