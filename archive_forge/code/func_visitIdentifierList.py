from antlr4 import *
def visitIdentifierList(self, ctx: fugue_sqlParser.IdentifierListContext):
    return self.visitChildren(ctx)