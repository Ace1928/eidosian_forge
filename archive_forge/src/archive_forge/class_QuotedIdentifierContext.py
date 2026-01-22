from antlr4 import *
from io import StringIO
import sys
class QuotedIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def BACKQUOTED_IDENTIFIER(self):
        return self.getToken(fugue_sqlParser.BACKQUOTED_IDENTIFIER, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_quotedIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQuotedIdentifier'):
            return visitor.visitQuotedIdentifier(self)
        else:
            return visitor.visitChildren(self)