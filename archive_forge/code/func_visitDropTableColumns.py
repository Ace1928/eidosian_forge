from antlr4 import *
def visitDropTableColumns(self, ctx: fugue_sqlParser.DropTableColumnsContext):
    return self.visitChildren(ctx)