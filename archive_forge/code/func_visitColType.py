from antlr4 import *
def visitColType(self, ctx: fugue_sqlParser.ColTypeContext):
    return self.visitChildren(ctx)