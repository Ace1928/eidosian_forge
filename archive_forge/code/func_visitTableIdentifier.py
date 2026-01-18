from antlr4 import *
def visitTableIdentifier(self, ctx: fugue_sqlParser.TableIdentifierContext):
    return self.visitChildren(ctx)