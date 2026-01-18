from antlr4 import *
def visitCreateTableHeader(self, ctx: fugue_sqlParser.CreateTableHeaderContext):
    return self.visitChildren(ctx)