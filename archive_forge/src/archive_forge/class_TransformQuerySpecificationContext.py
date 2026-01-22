from antlr4 import *
from io import StringIO
import sys
class TransformQuerySpecificationContext(QuerySpecificationContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def transformClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.TransformClauseContext, 0)

    def optionalFromClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.OptionalFromClauseContext, 0)

    def whereClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTransformQuerySpecification'):
            return visitor.visitTransformQuerySpecification(self)
        else:
            return visitor.visitChildren(self)