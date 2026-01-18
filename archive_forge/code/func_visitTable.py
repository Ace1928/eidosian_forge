from antlr4 import *
def visitTable(self, ctx: fugue_sqlParser.TableContext):
    return self.visitChildren(ctx)