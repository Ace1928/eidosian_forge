from antlr4 import *
def visitSetTableSerDe(self, ctx: fugue_sqlParser.SetTableSerDeContext):
    return self.visitChildren(ctx)