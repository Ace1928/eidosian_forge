from antlr4 import *
def visitDescribeColName(self, ctx: fugue_sqlParser.DescribeColNameContext):
    return self.visitChildren(ctx)