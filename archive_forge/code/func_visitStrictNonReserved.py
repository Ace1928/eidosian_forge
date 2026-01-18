from antlr4 import *
def visitStrictNonReserved(self, ctx: fugue_sqlParser.StrictNonReservedContext):
    return self.visitChildren(ctx)