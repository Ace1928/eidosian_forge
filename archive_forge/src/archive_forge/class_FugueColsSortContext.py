from antlr4 import *
from io import StringIO
import sys
class FugueColsSortContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueColSort(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueColSortContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueColSortContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueColsSort

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueColsSort'):
            return visitor.visitFugueColsSort(self)
        else:
            return visitor.visitChildren(self)