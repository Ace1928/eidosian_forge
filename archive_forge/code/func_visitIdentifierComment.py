from antlr4 import *
def visitIdentifierComment(self, ctx: fugue_sqlParser.IdentifierCommentContext):
    return self.visitChildren(ctx)