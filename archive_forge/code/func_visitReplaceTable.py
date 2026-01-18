from antlr4 import *
def visitReplaceTable(self, ctx: fugue_sqlParser.ReplaceTableContext):
    return self.visitChildren(ctx)