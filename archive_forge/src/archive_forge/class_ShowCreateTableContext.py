from antlr4 import *
from io import StringIO
import sys
class ShowCreateTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def SERDE(self):
        return self.getToken(fugue_sqlParser.SERDE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowCreateTable'):
            return visitor.visitShowCreateTable(self)
        else:
            return visitor.visitChildren(self)