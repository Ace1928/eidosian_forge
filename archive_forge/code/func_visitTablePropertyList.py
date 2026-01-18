from antlr4 import *
def visitTablePropertyList(self, ctx: fugue_sqlParser.TablePropertyListContext):
    return self.visitChildren(ctx)