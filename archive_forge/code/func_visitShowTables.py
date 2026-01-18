from antlr4 import *
def visitShowTables(self, ctx: fugue_sqlParser.ShowTablesContext):
    return self.visitChildren(ctx)