from antlr4 import *
def visitMatchedAction(self, ctx: fugue_sqlParser.MatchedActionContext):
    return self.visitChildren(ctx)