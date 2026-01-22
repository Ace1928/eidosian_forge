from antlr4 import *
from io import StringIO
import sys
class FromStmtContext(QueryPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fromStatement(self):
        return self.getTypedRuleContext(fugue_sqlParser.FromStatementContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFromStmt'):
            return visitor.visitFromStmt(self)
        else:
            return visitor.visitChildren(self)