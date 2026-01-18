from antlr4 import *
def visitFugueSchemaListType(self, ctx: fugue_sqlParser.FugueSchemaListTypeContext):
    return self.visitChildren(ctx)