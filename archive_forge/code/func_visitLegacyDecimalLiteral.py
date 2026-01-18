from antlr4 import *
def visitLegacyDecimalLiteral(self, ctx: fugue_sqlParser.LegacyDecimalLiteralContext):
    return self.visitChildren(ctx)