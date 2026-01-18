from antlr4 import *
def visitErrorCapturingIdentifier(self, ctx: fugue_sqlParser.ErrorCapturingIdentifierContext):
    return self.visitChildren(ctx)