from antlr4 import *
def visitLast(self, ctx: fugue_sqlParser.LastContext):
    return self.visitChildren(ctx)