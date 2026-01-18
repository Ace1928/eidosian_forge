from antlr4 import *
def visitLocationSpec(self, ctx: fugue_sqlParser.LocationSpecContext):
    return self.visitChildren(ctx)