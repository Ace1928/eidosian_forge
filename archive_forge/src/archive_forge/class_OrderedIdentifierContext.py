from antlr4 import *
from io import StringIO
import sys
class OrderedIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.ident = None
        self.ordering = None

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def ASC(self):
        return self.getToken(fugue_sqlParser.ASC, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_orderedIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitOrderedIdentifier'):
            return visitor.visitOrderedIdentifier(self)
        else:
            return visitor.visitChildren(self)