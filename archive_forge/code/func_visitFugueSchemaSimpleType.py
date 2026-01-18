from antlr4 import *
def visitFugueSchemaSimpleType(self, ctx: fugue_sqlParser.FugueSchemaSimpleTypeContext):
    return self.visitChildren(ctx)