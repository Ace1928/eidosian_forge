from antlr4 import *
from io import StringIO
import sys
class FugueDataFrameSourceContext(FugueDataFrameContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def fugueDataFrameMember(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameMemberContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFrameSource'):
            return visitor.visitFugueDataFrameSource(self)
        else:
            return visitor.visitChildren(self)