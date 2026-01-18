from antlr4 import *
def visitUnsetTableProperties(self, ctx: fugue_sqlParser.UnsetTablePropertiesContext):
    return self.visitChildren(ctx)