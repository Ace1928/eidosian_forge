from antlr4 import *
from io import StringIO
import sys
class FromClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def relation(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.RelationContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.RelationContext, i)

    def lateralView(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

    def pivotClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.PivotClauseContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fromClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFromClause'):
            return visitor.visitFromClause(self)
        else:
            return visitor.visitChildren(self)