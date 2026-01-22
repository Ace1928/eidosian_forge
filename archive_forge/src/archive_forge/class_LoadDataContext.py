from antlr4 import *
from io import StringIO
import sys
class LoadDataContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.path = None
        self.copyFrom(ctx)

    def LOAD(self):
        return self.getToken(fugue_sqlParser.LOAD, 0)

    def DATA(self):
        return self.getToken(fugue_sqlParser.DATA, 0)

    def INPATH(self):
        return self.getToken(fugue_sqlParser.INPATH, 0)

    def INTO(self):
        return self.getToken(fugue_sqlParser.INTO, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def LOCAL(self):
        return self.getToken(fugue_sqlParser.LOCAL, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLoadData'):
            return visitor.visitLoadData(self)
        else:
            return visitor.visitChildren(self)