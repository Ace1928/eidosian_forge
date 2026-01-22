from antlr4 import *
from io import StringIO
import sys
class RenameTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.ifrom = None
        self.to = None
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def RENAME(self):
        return self.getToken(fugue_sqlParser.RENAME, 0)

    def TO(self):
        return self.getToken(fugue_sqlParser.TO, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRenameTable'):
            return visitor.visitRenameTable(self)
        else:
            return visitor.visitChildren(self)