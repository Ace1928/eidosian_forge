from antlr4 import *
def visitOrderedIdentifierList(self, ctx: fugue_sqlParser.OrderedIdentifierListContext):
    return self.visitChildren(ctx)