from antlr4 import *
def visitInsertOverwriteDir(self, ctx: fugue_sqlParser.InsertOverwriteDirContext):
    return self.visitChildren(ctx)