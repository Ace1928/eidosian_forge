from antlr4 import *
from io import StringIO
import sys
class InlineTableDefault2Context(RelationPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def inlineTable(self):
        return self.getTypedRuleContext(fugue_sqlParser.InlineTableContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInlineTableDefault2'):
            return visitor.visitInlineTableDefault2(self)
        else:
            return visitor.visitChildren(self)