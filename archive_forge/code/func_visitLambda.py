from antlr4 import *
def visitLambda(self, ctx: fugue_sqlParser.LambdaContext):
    return self.visitChildren(ctx)