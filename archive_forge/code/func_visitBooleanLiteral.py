from antlr4 import *
def visitBooleanLiteral(self, ctx: fugue_sqlParser.BooleanLiteralContext):
    return self.visitChildren(ctx)