from antlr4 import *
from io import StringIO
import sys
class CreateFunctionContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.className = None
        self.copyFrom(ctx)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def FUNCTION(self):
        return self.getToken(fugue_sqlParser.FUNCTION, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def OR(self):
        return self.getToken(fugue_sqlParser.OR, 0)

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def TEMPORARY(self):
        return self.getToken(fugue_sqlParser.TEMPORARY, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def resource(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ResourceContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ResourceContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateFunction'):
            return visitor.visitCreateFunction(self)
        else:
            return visitor.visitChildren(self)