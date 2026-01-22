from antlr4 import *
from io import StringIO
import sys
class NestedConstantListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def constantList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ConstantListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ConstantListContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_nestedConstantList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNestedConstantList'):
            return visitor.visitNestedConstantList(self)
        else:
            return visitor.visitChildren(self)