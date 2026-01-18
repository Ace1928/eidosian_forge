from antlr4 import *
def visitExtract(self, ctx: fugue_sqlParser.ExtractContext):
    return self.visitChildren(ctx)