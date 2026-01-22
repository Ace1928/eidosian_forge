from antlr4 import *
from io import StringIO
import sys
class QueryPrimaryDefaultContext(QueryPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def querySpecification(self):
        return self.getTypedRuleContext(fugue_sqlParser.QuerySpecificationContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQueryPrimaryDefault'):
            return visitor.visitQueryPrimaryDefault(self)
        else:
            return visitor.visitChildren(self)