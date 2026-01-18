from antlr4 import *
def visitDecimalLiteral(self, ctx: fugue_sqlParser.DecimalLiteralContext):
    return self.visitChildren(ctx)