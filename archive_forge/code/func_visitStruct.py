from antlr4 import *
def visitStruct(self, ctx: fugue_sqlParser.StructContext):
    return self.visitChildren(ctx)