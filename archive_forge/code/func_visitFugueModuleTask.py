from antlr4 import *
def visitFugueModuleTask(self, ctx: fugue_sqlParser.FugueModuleTaskContext):
    return self.visitChildren(ctx)