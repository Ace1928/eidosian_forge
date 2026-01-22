from antlr4 import *
from io import StringIO
import sys
class FugueBroadcastContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def BROADCAST(self):
        return self.getToken(fugue_sqlParser.BROADCAST, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueBroadcast

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueBroadcast'):
            return visitor.visitFugueBroadcast(self)
        else:
            return visitor.visitChildren(self)