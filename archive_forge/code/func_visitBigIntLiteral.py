from antlr4 import *
def visitBigIntLiteral(self, ctx: fugue_sqlParser.BigIntLiteralContext):
    return self.visitChildren(ctx)