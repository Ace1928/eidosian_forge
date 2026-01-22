from antlr4 import *
from io import StringIO
import sys
class FugueTermContext(QueryTermContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueNestableTaskCollectionNoSelect(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskCollectionNoSelectContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueTerm'):
            return visitor.visitFugueTerm(self)
        else:
            return visitor.visitChildren(self)