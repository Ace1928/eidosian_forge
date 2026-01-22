from antlr4 import *
from io import StringIO
import sys
class ExtractContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.field = None
        self.source = None
        self.copyFrom(ctx)

    def EXTRACT(self):
        return self.getToken(fugue_sqlParser.EXTRACT, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def valueExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitExtract'):
            return visitor.visitExtract(self)
        else:
            return visitor.visitChildren(self)