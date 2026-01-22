from antlr4 import *
from io import StringIO
import sys
class ShowFunctionsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.pattern = None
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def FUNCTIONS(self):
        return self.getToken(fugue_sqlParser.FUNCTIONS, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def LIKE(self):
        return self.getToken(fugue_sqlParser.LIKE, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowFunctions'):
            return visitor.visitShowFunctions(self)
        else:
            return visitor.visitChildren(self)