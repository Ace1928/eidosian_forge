from antlr4 import *
def visitDescribeFunction(self, ctx: fugue_sqlParser.DescribeFunctionContext):
    return self.visitChildren(ctx)