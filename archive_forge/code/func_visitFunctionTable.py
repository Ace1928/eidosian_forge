from antlr4 import *
def visitFunctionTable(self, ctx: fugue_sqlParser.FunctionTableContext):
    return self.visitChildren(ctx)