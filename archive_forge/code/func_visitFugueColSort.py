from antlr4 import *
def visitFugueColSort(self, ctx: fugue_sqlParser.FugueColSortContext):
    return self.visitChildren(ctx)