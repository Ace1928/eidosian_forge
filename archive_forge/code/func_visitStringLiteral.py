from antlr4 import *
def visitStringLiteral(self, ctx: fugue_sqlParser.StringLiteralContext):
    return self.visitChildren(ctx)