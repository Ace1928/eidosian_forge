from antlr4 import *
def visitIntervalValue(self, ctx: fugue_sqlParser.IntervalValueContext):
    return self.visitChildren(ctx)