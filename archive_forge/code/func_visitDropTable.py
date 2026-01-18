from antlr4 import *
def visitDropTable(self, ctx: fugue_sqlParser.DropTableContext):
    return self.visitChildren(ctx)