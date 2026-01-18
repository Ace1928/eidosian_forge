from antlr4 import *
def visitFugueCreateDataTask(self, ctx: fugue_sqlParser.FugueCreateDataTaskContext):
    return self.visitChildren(ctx)