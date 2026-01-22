from antlr4 import *
from io import StringIO
import sys
class ShowTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.ns = None
        self.pattern = None
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def EXTENDED(self):
        return self.getToken(fugue_sqlParser.EXTENDED, 0)

    def LIKE(self):
        return self.getToken(fugue_sqlParser.LIKE, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def IN(self):
        return self.getToken(fugue_sqlParser.IN, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowTable'):
            return visitor.visitShowTable(self)
        else:
            return visitor.visitChildren(self)