from antlr4 import *
from io import StringIO
import sys
class JoinTypeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def INNER(self):
        return self.getToken(fugue_sqlParser.INNER, 0)

    def CROSS(self):
        return self.getToken(fugue_sqlParser.CROSS, 0)

    def LEFT(self):
        return self.getToken(fugue_sqlParser.LEFT, 0)

    def OUTER(self):
        return self.getToken(fugue_sqlParser.OUTER, 0)

    def SEMI(self):
        return self.getToken(fugue_sqlParser.SEMI, 0)

    def RIGHT(self):
        return self.getToken(fugue_sqlParser.RIGHT, 0)

    def FULL(self):
        return self.getToken(fugue_sqlParser.FULL, 0)

    def ANTI(self):
        return self.getToken(fugue_sqlParser.ANTI, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_joinType

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitJoinType'):
            return visitor.visitJoinType(self)
        else:
            return visitor.visitChildren(self)