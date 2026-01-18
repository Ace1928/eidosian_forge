from antlr4 import *
def visitNonReserved(self, ctx: fugue_sqlParser.NonReservedContext):
    return self.visitChildren(ctx)