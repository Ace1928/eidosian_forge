from antlr4 import *
from io import StringIO
import sys
class RegularQuerySpecificationContext(QuerySpecificationContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def selectClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.SelectClauseContext, 0)

    def optionalFromClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.OptionalFromClauseContext, 0)

    def lateralView(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

    def whereClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

    def aggregationClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.AggregationClauseContext, 0)

    def havingClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.HavingClauseContext, 0)

    def windowClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRegularQuerySpecification'):
            return visitor.visitRegularQuerySpecification(self)
        else:
            return visitor.visitChildren(self)