from antlr4 import *
def visitSortItem(self, ctx: fugue_sqlParser.SortItemContext):
    return self.visitChildren(ctx)