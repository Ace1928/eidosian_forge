from antlr4 import *
def visitFugueAssignment(self, ctx: fugue_sqlParser.FugueAssignmentContext):
    return self.visitChildren(ctx)