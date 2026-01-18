from antlr4 import *
def visitAddTableColumns(self, ctx: fugue_sqlParser.AddTableColumnsContext):
    return self.visitChildren(ctx)