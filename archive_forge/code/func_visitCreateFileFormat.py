from antlr4 import *
def visitCreateFileFormat(self, ctx: fugue_sqlParser.CreateFileFormatContext):
    return self.visitChildren(ctx)