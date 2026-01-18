from antlr4 import *
def visitSubstring(self, ctx: fugue_sqlParser.SubstringContext):
    return self.visitChildren(ctx)