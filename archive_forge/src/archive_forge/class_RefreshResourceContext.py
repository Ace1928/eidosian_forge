from antlr4 import *
from io import StringIO
import sys
class RefreshResourceContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def REFRESH(self):
        return self.getToken(fugue_sqlParser.REFRESH, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRefreshResource'):
            return visitor.visitRefreshResource(self)
        else:
            return visitor.visitChildren(self)