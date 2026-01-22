from antlr4 import *
from io import StringIO
import sys
class JoinRelationContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.right = None

    def joinType(self):
        return self.getTypedRuleContext(fugue_sqlParser.JoinTypeContext, 0)

    def JOIN(self):
        return self.getToken(fugue_sqlParser.JOIN, 0)

    def relationPrimary(self):
        return self.getTypedRuleContext(fugue_sqlParser.RelationPrimaryContext, 0)

    def joinCriteria(self):
        return self.getTypedRuleContext(fugue_sqlParser.JoinCriteriaContext, 0)

    def NATURAL(self):
        return self.getToken(fugue_sqlParser.NATURAL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_joinRelation

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitJoinRelation'):
            return visitor.visitJoinRelation(self)
        else:
            return visitor.visitChildren(self)