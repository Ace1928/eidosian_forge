from antlr4 import *
def visitExponentLiteral(self, ctx: fugue_sqlParser.ExponentLiteralContext):
    return self.visitChildren(ctx)