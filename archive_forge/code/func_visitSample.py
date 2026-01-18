from antlr4 import *
def visitSample(self, ctx: fugue_sqlParser.SampleContext):
    return self.visitChildren(ctx)