from antlr4 import *
from io import StringIO
import sys
class RowFormatSerdeContext(RowFormatContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.name = None
        self.props = None
        self.copyFrom(ctx)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def FORMAT(self):
        return self.getToken(fugue_sqlParser.FORMAT, 0)

    def SERDE(self):
        return self.getToken(fugue_sqlParser.SERDE, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def WITH(self):
        return self.getToken(fugue_sqlParser.WITH, 0)

    def SERDEPROPERTIES(self):
        return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

    def tablePropertyList(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRowFormatSerde'):
            return visitor.visitRowFormatSerde(self)
        else:
            return visitor.visitChildren(self)