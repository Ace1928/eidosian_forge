from antlr4 import *
from io import StringIO
import sys
class ExplainContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def EXPLAIN(self):
        return self.getToken(fugue_sqlParser.EXPLAIN, 0)

    def statement(self):
        return self.getTypedRuleContext(fugue_sqlParser.StatementContext, 0)

    def LOGICAL(self):
        return self.getToken(fugue_sqlParser.LOGICAL, 0)

    def FORMATTED(self):
        return self.getToken(fugue_sqlParser.FORMATTED, 0)

    def EXTENDED(self):
        return self.getToken(fugue_sqlParser.EXTENDED, 0)

    def CODEGEN(self):
        return self.getToken(fugue_sqlParser.CODEGEN, 0)

    def COST(self):
        return self.getToken(fugue_sqlParser.COST, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitExplain'):
            return visitor.visitExplain(self)
        else:
            return visitor.visitChildren(self)