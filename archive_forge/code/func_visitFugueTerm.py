from antlr4 import *
def visitFugueTerm(self, ctx: fugue_sqlParser.FugueTermContext):
    return self.visitChildren(ctx)