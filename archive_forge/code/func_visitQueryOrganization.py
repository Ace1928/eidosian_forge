from antlr4 import *
def visitQueryOrganization(self, ctx: fugue_sqlParser.QueryOrganizationContext):
    return self.visitChildren(ctx)