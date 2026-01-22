from antlr4 import *
from io import StringIO
import sys
class AliasedRelationContext(RelationPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def relation(self):
        return self.getTypedRuleContext(fugue_sqlParser.RelationContext, 0)

    def tableAlias(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

    def sample(self):
        return self.getTypedRuleContext(fugue_sqlParser.SampleContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAliasedRelation'):
            return visitor.visitAliasedRelation(self)
        else:
            return visitor.visitChildren(self)