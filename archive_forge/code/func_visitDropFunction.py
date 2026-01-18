from antlr4 import *
def visitDropFunction(self, ctx: fugue_sqlParser.DropFunctionContext):
    return self.visitChildren(ctx)