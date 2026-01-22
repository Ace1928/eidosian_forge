from antlr4 import *
from io import StringIO
import sys
class SingleDataTypeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def dataType(self):
        return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_singleDataType

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSingleDataType'):
            return visitor.visitSingleDataType(self)
        else:
            return visitor.visitChildren(self)