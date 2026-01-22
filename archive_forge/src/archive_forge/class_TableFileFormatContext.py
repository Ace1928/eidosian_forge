from antlr4 import *
from io import StringIO
import sys
class TableFileFormatContext(FileFormatContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.inFmt = None
        self.outFmt = None
        self.copyFrom(ctx)

    def INPUTFORMAT(self):
        return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)

    def OUTPUTFORMAT(self):
        return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)

    def STRING(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.STRING)
        else:
            return self.getToken(fugue_sqlParser.STRING, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableFileFormat'):
            return visitor.visitTableFileFormat(self)
        else:
            return visitor.visitChildren(self)