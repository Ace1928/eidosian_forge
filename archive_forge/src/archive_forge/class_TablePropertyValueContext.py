from antlr4 import *
from io import StringIO
import sys
class TablePropertyValueContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def booleanValue(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanValueContext, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tablePropertyValue

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTablePropertyValue'):
            return visitor.visitTablePropertyValue(self)
        else:
            return visitor.visitChildren(self)