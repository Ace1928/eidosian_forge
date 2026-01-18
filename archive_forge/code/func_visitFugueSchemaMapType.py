from antlr4 import *
def visitFugueSchemaMapType(self, ctx: fugue_sqlParser.FugueSchemaMapTypeContext):
    return self.visitChildren(ctx)