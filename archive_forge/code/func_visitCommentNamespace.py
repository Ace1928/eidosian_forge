from antlr4 import *
def visitCommentNamespace(self, ctx: fugue_sqlParser.CommentNamespaceContext):
    return self.visitChildren(ctx)