from antlr4 import *
def visitUnsupportedHiveNativeCommands(self, ctx: fugue_sqlParser.UnsupportedHiveNativeCommandsContext):
    return self.visitChildren(ctx)