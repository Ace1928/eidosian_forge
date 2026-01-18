from antlr4 import *
def visitCreateView(self, ctx: fugue_sqlParser.CreateViewContext):
    return self.visitChildren(ctx)