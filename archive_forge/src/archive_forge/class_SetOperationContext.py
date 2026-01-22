from antlr4 import *
from io import StringIO
import sys
class SetOperationContext(QueryTermContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.left = None
        self.theOperator = None
        self.right = None
        self.copyFrom(ctx)

    def queryTerm(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.QueryTermContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.QueryTermContext, i)

    def INTERSECT(self):
        return self.getToken(fugue_sqlParser.INTERSECT, 0)

    def UNION(self):
        return self.getToken(fugue_sqlParser.UNION, 0)

    def EXCEPT(self):
        return self.getToken(fugue_sqlParser.EXCEPT, 0)

    def SETMINUS(self):
        return self.getToken(fugue_sqlParser.SETMINUS, 0)

    def setQuantifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.SetQuantifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetOperation'):
            return visitor.visitSetOperation(self)
        else:
            return visitor.visitChildren(self)