from antlr4 import *
def visitUpdateTable(self, ctx: fugue_sqlParser.UpdateTableContext):
    return self.visitChildren(ctx)