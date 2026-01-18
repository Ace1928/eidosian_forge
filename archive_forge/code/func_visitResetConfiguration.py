from antlr4 import *
def visitResetConfiguration(self, ctx: fugue_sqlParser.ResetConfigurationContext):
    return self.visitChildren(ctx)