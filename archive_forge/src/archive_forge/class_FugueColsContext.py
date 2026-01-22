from antlr4 import *
from io import StringIO
import sys
class FugueColsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueColumnIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueColumnIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueColumnIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueCols

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueCols'):
            return visitor.visitFugueCols(self)
        else:
            return visitor.visitChildren(self)