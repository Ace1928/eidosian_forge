from antlr4 import *
from io import StringIO
import sys
class QueryTermDefaultContext(QueryTermContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def queryPrimary(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryPrimaryContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQueryTermDefault'):
            return visitor.visitQueryTermDefault(self)
        else:
            return visitor.visitChildren(self)