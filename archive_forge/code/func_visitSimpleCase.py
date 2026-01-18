from antlr4 import *
def visitSimpleCase(self, ctx: fugue_sqlParser.SimpleCaseContext):
    return self.visitChildren(ctx)