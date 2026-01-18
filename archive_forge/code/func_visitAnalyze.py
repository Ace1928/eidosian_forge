from antlr4 import *
def visitAnalyze(self, ctx: fugue_sqlParser.AnalyzeContext):
    return self.visitChildren(ctx)