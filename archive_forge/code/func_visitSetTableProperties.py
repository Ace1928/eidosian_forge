from antlr4 import *
def visitSetTableProperties(self, ctx: fugue_sqlParser.SetTablePropertiesContext):
    return self.visitChildren(ctx)