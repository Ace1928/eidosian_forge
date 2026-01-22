from antlr4 import *
from io import StringIO
import sys
class AlterViewQueryContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAlterViewQuery'):
            return visitor.visitAlterViewQuery(self)
        else:
            return visitor.visitChildren(self)