from antlr4 import *
def visitFugueJsonBool(self, ctx: fugue_sqlParser.FugueJsonBoolContext):
    return self.visitChildren(ctx)