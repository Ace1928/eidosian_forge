from antlr4 import *
def visitTableFileFormat(self, ctx: fugue_sqlParser.TableFileFormatContext):
    return self.visitChildren(ctx)