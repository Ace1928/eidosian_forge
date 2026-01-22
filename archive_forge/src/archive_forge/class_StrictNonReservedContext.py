from antlr4 import *
from io import StringIO
import sys
class StrictNonReservedContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ANTI(self):
        return self.getToken(fugue_sqlParser.ANTI, 0)

    def CROSS(self):
        return self.getToken(fugue_sqlParser.CROSS, 0)

    def EXCEPT(self):
        return self.getToken(fugue_sqlParser.EXCEPT, 0)

    def FULL(self):
        return self.getToken(fugue_sqlParser.FULL, 0)

    def INNER(self):
        return self.getToken(fugue_sqlParser.INNER, 0)

    def INTERSECT(self):
        return self.getToken(fugue_sqlParser.INTERSECT, 0)

    def JOIN(self):
        return self.getToken(fugue_sqlParser.JOIN, 0)

    def LEFT(self):
        return self.getToken(fugue_sqlParser.LEFT, 0)

    def NATURAL(self):
        return self.getToken(fugue_sqlParser.NATURAL, 0)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def RIGHT(self):
        return self.getToken(fugue_sqlParser.RIGHT, 0)

    def SEMI(self):
        return self.getToken(fugue_sqlParser.SEMI, 0)

    def SETMINUS(self):
        return self.getToken(fugue_sqlParser.SETMINUS, 0)

    def UNION(self):
        return self.getToken(fugue_sqlParser.UNION, 0)

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_strictNonReserved

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStrictNonReserved'):
            return visitor.visitStrictNonReserved(self)
        else:
            return visitor.visitChildren(self)