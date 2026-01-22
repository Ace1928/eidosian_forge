from antlr4 import *
from io import StringIO
import sys
class FugueSchemaListTypeContext(FugueSchemaTypeContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueSchemaType(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchemaListType'):
            return visitor.visitFugueSchemaListType(self)
        else:
            return visitor.visitChildren(self)