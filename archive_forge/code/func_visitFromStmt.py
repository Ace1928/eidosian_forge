from antlr4 import *
def visitFromStmt(self, ctx: fugue_sqlParser.FromStmtContext):
    return self.visitChildren(ctx)