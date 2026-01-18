from antlr4 import *
def visitArithmeticBinary(self, ctx: fugue_sqlParser.ArithmeticBinaryContext):
    return self.visitChildren(ctx)