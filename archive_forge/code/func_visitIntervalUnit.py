from antlr4 import *
def visitIntervalUnit(self, ctx: fugue_sqlParser.IntervalUnitContext):
    return self.visitChildren(ctx)