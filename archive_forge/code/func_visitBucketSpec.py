from antlr4 import *
def visitBucketSpec(self, ctx: fugue_sqlParser.BucketSpecContext):
    return self.visitChildren(ctx)