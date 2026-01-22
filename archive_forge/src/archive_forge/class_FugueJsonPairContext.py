from antlr4 import *
from io import StringIO
import sys
class FugueJsonPairContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.key = None
        self.value = None

    def fugueJsonKey(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonKeyContext, 0)

    def EQUAL(self):
        return self.getToken(fugue_sqlParser.EQUAL, 0)

    def fugueJsonValue(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonPair

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonPair'):
            return visitor.visitFugueJsonPair(self)
        else:
            return visitor.visitChildren(self)