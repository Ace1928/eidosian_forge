from antlr4 import *
def visitNamedWindow(self, ctx: fugue_sqlParser.NamedWindowContext):
    return self.visitChildren(ctx)