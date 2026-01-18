from antlr4 import *
def visitFugueLanguage(self, ctx: fugue_sqlParser.FugueLanguageContext):
    return self.visitChildren(ctx)