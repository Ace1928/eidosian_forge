from antlr4 import *
def visitSetQuantifier(self, ctx: fugue_sqlParser.SetQuantifierContext):
    return self.visitChildren(ctx)