from antlr4 import *
from io import StringIO
import sys
class SubstringContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.istr = None
        self.pos = None
        self.ilen = None
        self.copyFrom(ctx)

    def SUBSTR(self):
        return self.getToken(fugue_sqlParser.SUBSTR, 0)

    def SUBSTRING(self):
        return self.getToken(fugue_sqlParser.SUBSTRING, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def FOR(self):
        return self.getToken(fugue_sqlParser.FOR, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSubstring'):
            return visitor.visitSubstring(self)
        else:
            return visitor.visitChildren(self)