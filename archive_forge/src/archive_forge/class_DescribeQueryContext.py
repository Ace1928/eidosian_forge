from antlr4 import *
from io import StringIO
import sys
class DescribeQueryContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def DESCRIBE(self):
        return self.getToken(fugue_sqlParser.DESCRIBE, 0)

    def QUERY(self):
        return self.getToken(fugue_sqlParser.QUERY, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDescribeQuery'):
            return visitor.visitDescribeQuery(self)
        else:
            return visitor.visitChildren(self)