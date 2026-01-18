from antlr4 import *
def visitRepairTable(self, ctx: fugue_sqlParser.RepairTableContext):
    return self.visitChildren(ctx)