from antlr4 import *
def visitFugueAssignmentSign(self, ctx: fugue_sqlParser.FugueAssignmentSignContext):
    return self.visitChildren(ctx)