from antlr4 import *
def visitSetConfiguration(self, ctx: fugue_sqlParser.SetConfigurationContext):
    return self.visitChildren(ctx)