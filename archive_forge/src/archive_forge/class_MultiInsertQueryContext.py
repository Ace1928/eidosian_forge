from antlr4 import *
from io import StringIO
import sys
class MultiInsertQueryContext(DmlStatementNoWithContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fromClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

    def multiInsertQueryBody(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultiInsertQueryBodyContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultiInsertQueryBodyContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMultiInsertQuery'):
            return visitor.visitMultiInsertQuery(self)
        else:
            return visitor.visitChildren(self)