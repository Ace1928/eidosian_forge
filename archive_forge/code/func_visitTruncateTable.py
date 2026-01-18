from antlr4 import *
def visitTruncateTable(self, ctx: fugue_sqlParser.TruncateTableContext):
    return self.visitChildren(ctx)