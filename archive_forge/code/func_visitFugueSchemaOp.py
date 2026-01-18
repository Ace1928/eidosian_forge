from antlr4 import *
def visitFugueSchemaOp(self, ctx: fugue_sqlParser.FugueSchemaOpContext):
    return self.visitChildren(ctx)