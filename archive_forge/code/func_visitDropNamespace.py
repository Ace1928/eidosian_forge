from antlr4 import *
def visitDropNamespace(self, ctx: fugue_sqlParser.DropNamespaceContext):
    return self.visitChildren(ctx)