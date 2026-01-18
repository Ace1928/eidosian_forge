from antlr4 import *
def visitTablePropertyValue(self, ctx: fugue_sqlParser.TablePropertyValueContext):
    return self.visitChildren(ctx)