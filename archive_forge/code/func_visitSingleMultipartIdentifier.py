from antlr4 import *
def visitSingleMultipartIdentifier(self, ctx: fugue_sqlParser.SingleMultipartIdentifierContext):
    return self.visitChildren(ctx)