from antlr4 import *
def visitIdentifier(self, ctx: fugue_sqlParser.IdentifierContext):
    return self.visitChildren(ctx)