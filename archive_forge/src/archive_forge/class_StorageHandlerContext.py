from antlr4 import *
from io import StringIO
import sys
class StorageHandlerContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def WITH(self):
        return self.getToken(fugue_sqlParser.WITH, 0)

    def SERDEPROPERTIES(self):
        return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

    def tablePropertyList(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_storageHandler

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStorageHandler'):
            return visitor.visitStorageHandler(self)
        else:
            return visitor.visitChildren(self)