from antlr4 import *
def visitFugueFileFormat(self, ctx: fugue_sqlParser.FugueFileFormatContext):
    return self.visitChildren(ctx)