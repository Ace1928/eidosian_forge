from antlr4 import *
from io import StringIO
import sys
class MergeIntoTableContext(DmlStatementNoWithContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.target = None
        self.targetAlias = None
        self.source = None
        self.sourceQuery = None
        self.sourceAlias = None
        self.mergeCondition = None
        self.copyFrom(ctx)

    def MERGE(self):
        return self.getToken(fugue_sqlParser.MERGE, 0)

    def INTO(self):
        return self.getToken(fugue_sqlParser.INTO, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def tableAlias(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TableAliasContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, i)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def matchedClause(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MatchedClauseContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MatchedClauseContext, i)

    def notMatchedClause(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NotMatchedClauseContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NotMatchedClauseContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMergeIntoTable'):
            return visitor.visitMergeIntoTable(self)
        else:
            return visitor.visitChildren(self)