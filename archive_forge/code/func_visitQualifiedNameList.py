from antlr4 import *
def visitQualifiedNameList(self, ctx: fugue_sqlParser.QualifiedNameListContext):
    return self.visitChildren(ctx)