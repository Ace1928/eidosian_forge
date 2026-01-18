from antlr4 import *
def visitDereference(self, ctx: fugue_sqlParser.DereferenceContext):
    return self.visitChildren(ctx)