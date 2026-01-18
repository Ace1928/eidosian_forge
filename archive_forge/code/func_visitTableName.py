from antlr4 import *
def visitTableName(self, ctx: fugue_sqlParser.TableNameContext):
    return self.visitChildren(ctx)