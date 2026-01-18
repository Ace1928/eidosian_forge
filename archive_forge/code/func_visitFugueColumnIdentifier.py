from antlr4 import *
def visitFugueColumnIdentifier(self, ctx: fugue_sqlParser.FugueColumnIdentifierContext):
    return self.visitChildren(ctx)