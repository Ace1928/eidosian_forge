from antlr4 import *
def visitMergeIntoTable(self, ctx: fugue_sqlParser.MergeIntoTableContext):
    return self.visitChildren(ctx)