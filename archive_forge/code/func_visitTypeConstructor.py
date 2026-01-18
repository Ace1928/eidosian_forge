from antlr4 import *
def visitTypeConstructor(self, ctx: fugue_sqlParser.TypeConstructorContext):
    return self.visitChildren(ctx)