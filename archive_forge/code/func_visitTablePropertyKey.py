from antlr4 import *
def visitTablePropertyKey(self, ctx: fugue_sqlParser.TablePropertyKeyContext):
    return self.visitChildren(ctx)