from antlr4 import *
def visitFunctionIdentifier(self, ctx: fugue_sqlParser.FunctionIdentifierContext):
    return self.visitChildren(ctx)