from antlr4 import *
def visitAssignment(self, ctx: fugue_sqlParser.AssignmentContext):
    return self.visitChildren(ctx)