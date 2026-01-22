from antlr4 import *
from io import StringIO
import sys
class IdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def strictIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.StrictIdentifierContext, 0)

    def strictNonReserved(self):
        return self.getTypedRuleContext(fugue_sqlParser.StrictNonReservedContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_identifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIdentifier'):
            return visitor.visitIdentifier(self)
        else:
            return visitor.visitChildren(self)