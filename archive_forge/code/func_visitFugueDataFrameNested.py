from antlr4 import *
def visitFugueDataFrameNested(self, ctx: fugue_sqlParser.FugueDataFrameNestedContext):
    return self.visitChildren(ctx)