from antlr4 import *
def visitMultipartIdentifierList(self, ctx: fugue_sqlParser.MultipartIdentifierListContext):
    return self.visitChildren(ctx)