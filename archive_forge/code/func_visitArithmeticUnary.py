from antlr4 import *
def visitArithmeticUnary(self, ctx: fugue_sqlParser.ArithmeticUnaryContext):
    return self.visitChildren(ctx)