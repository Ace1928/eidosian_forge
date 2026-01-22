from antlr4 import *
from io import StringIO
import sys
class SortItemContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.ordering = None
        self.nullOrder = None

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def ASC(self):
        return self.getToken(fugue_sqlParser.ASC, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def LAST(self):
        return self.getToken(fugue_sqlParser.LAST, 0)

    def FIRST(self):
        return self.getToken(fugue_sqlParser.FIRST, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_sortItem

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSortItem'):
            return visitor.visitSortItem(self)
        else:
            return visitor.visitChildren(self)