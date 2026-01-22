from antlr4 import *
from io import StringIO
import sys
class TableNameContext(RelationPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def tableAlias(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

    def fugueDataFrameMember(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameMemberContext, 0)

    def sample(self):
        return self.getTypedRuleContext(fugue_sqlParser.SampleContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableName'):
            return visitor.visitTableName(self)
        else:
            return visitor.visitChildren(self)