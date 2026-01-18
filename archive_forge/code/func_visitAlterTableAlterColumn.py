from antlr4 import *
def visitAlterTableAlterColumn(self, ctx: fugue_sqlParser.AlterTableAlterColumnContext):
    return self.visitChildren(ctx)