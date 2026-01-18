from antlr4 import *
def visitCreateTable(self, ctx: fugue_sqlParser.CreateTableContext):
    return self.visitChildren(ctx)