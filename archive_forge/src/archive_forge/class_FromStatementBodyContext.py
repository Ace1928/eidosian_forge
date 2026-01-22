from antlr4 import *
from io import StringIO
import sys
class FromStatementBodyContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def transformClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.TransformClauseContext, 0)

    def queryOrganization(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryOrganizationContext, 0)

    def whereClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

    def selectClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.SelectClauseContext, 0)

    def lateralView(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

    def aggregationClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.AggregationClauseContext, 0)

    def havingClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.HavingClauseContext, 0)

    def windowClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fromStatementBody

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFromStatementBody'):
            return visitor.visitFromStatementBody(self)
        else:
            return visitor.visitChildren(self)