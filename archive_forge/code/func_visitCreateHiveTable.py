from antlr4 import *
def visitCreateHiveTable(self, ctx: fugue_sqlParser.CreateHiveTableContext):
    return self.visitChildren(ctx)