from antlr4 import *
def visitDeleteFromTable(self, ctx: fugue_sqlParser.DeleteFromTableContext):
    return self.visitChildren(ctx)