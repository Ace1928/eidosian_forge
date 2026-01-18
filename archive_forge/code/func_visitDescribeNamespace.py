from antlr4 import *
def visitDescribeNamespace(self, ctx: fugue_sqlParser.DescribeNamespaceContext):
    return self.visitChildren(ctx)