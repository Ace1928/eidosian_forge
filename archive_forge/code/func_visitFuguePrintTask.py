from antlr4 import *
def visitFuguePrintTask(self, ctx: fugue_sqlParser.FuguePrintTaskContext):
    return self.visitChildren(ctx)