from antlr4 import *
from io import StringIO
import sys
class SetQuantifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DISTINCT(self):
        return self.getToken(fugue_sqlParser.DISTINCT, 0)

    def ALL(self):
        return self.getToken(fugue_sqlParser.ALL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_setQuantifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetQuantifier'):
            return visitor.visitSetQuantifier(self)
        else:
            return visitor.visitChildren(self)