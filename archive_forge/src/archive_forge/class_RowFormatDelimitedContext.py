from antlr4 import *
from io import StringIO
import sys
class RowFormatDelimitedContext(RowFormatContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.fieldsTerminatedBy = None
        self.escapedBy = None
        self.collectionItemsTerminatedBy = None
        self.keysTerminatedBy = None
        self.linesSeparatedBy = None
        self.nullDefinedAs = None
        self.copyFrom(ctx)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def FORMAT(self):
        return self.getToken(fugue_sqlParser.FORMAT, 0)

    def DELIMITED(self):
        return self.getToken(fugue_sqlParser.DELIMITED, 0)

    def FIELDS(self):
        return self.getToken(fugue_sqlParser.FIELDS, 0)

    def TERMINATED(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.TERMINATED)
        else:
            return self.getToken(fugue_sqlParser.TERMINATED, i)

    def BY(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.BY)
        else:
            return self.getToken(fugue_sqlParser.BY, i)

    def COLLECTION(self):
        return self.getToken(fugue_sqlParser.COLLECTION, 0)

    def ITEMS(self):
        return self.getToken(fugue_sqlParser.ITEMS, 0)

    def MAP(self):
        return self.getToken(fugue_sqlParser.MAP, 0)

    def KEYS(self):
        return self.getToken(fugue_sqlParser.KEYS, 0)

    def LINES(self):
        return self.getToken(fugue_sqlParser.LINES, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def DEFINED(self):
        return self.getToken(fugue_sqlParser.DEFINED, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def STRING(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.STRING)
        else:
            return self.getToken(fugue_sqlParser.STRING, i)

    def ESCAPED(self):
        return self.getToken(fugue_sqlParser.ESCAPED, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRowFormatDelimited'):
            return visitor.visitRowFormatDelimited(self)
        else:
            return visitor.visitChildren(self)