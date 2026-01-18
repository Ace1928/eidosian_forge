from antlr4 import *
def visitFugueDataFrameMember(self, ctx: fugue_sqlParser.FugueDataFrameMemberContext):
    return self.visitChildren(ctx)