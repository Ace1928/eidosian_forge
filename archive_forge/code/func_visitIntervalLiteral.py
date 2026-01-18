from antlr4 import *
def visitIntervalLiteral(self, ctx: fugue_sqlParser.IntervalLiteralContext):
    return self.visitChildren(ctx)