from antlr4 import *
def visitInlineTable(self, ctx: fugue_sqlParser.InlineTableContext):
    return self.visitChildren(ctx)