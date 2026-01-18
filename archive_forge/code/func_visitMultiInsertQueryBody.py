from antlr4 import *
def visitMultiInsertQueryBody(self, ctx: fugue_sqlParser.MultiInsertQueryBodyContext):
    return self.visitChildren(ctx)