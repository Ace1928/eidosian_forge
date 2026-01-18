from antlr4 import *
def visitCommentSpec(self, ctx: fugue_sqlParser.CommentSpecContext):
    return self.visitChildren(ctx)