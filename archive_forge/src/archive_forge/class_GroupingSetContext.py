from antlr4 import *
from io import StringIO
import sys
class GroupingSetContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_groupingSet

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitGroupingSet'):
            return visitor.visitGroupingSet(self)
        else:
            return visitor.visitChildren(self)