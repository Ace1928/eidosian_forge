from antlr4 import *
from io import StringIO
import sys
class OrderedIdentifierListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def orderedIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.OrderedIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.OrderedIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_orderedIdentifierList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitOrderedIdentifierList'):
            return visitor.visitOrderedIdentifierList(self)
        else:
            return visitor.visitChildren(self)