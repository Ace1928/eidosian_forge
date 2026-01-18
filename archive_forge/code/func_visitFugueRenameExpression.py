from antlr4 import *
def visitFugueRenameExpression(self, ctx: fugue_sqlParser.FugueRenameExpressionContext):
    return self.visitChildren(ctx)