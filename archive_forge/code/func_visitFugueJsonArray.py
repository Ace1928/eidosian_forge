from antlr4 import *
def visitFugueJsonArray(self, ctx: fugue_sqlParser.FugueJsonArrayContext):
    return self.visitChildren(ctx)