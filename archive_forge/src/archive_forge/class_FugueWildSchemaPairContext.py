from antlr4 import *
from io import StringIO
import sys
class FugueWildSchemaPairContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.pair = None

    def fugueSchemaPair(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaPairContext, 0)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueWildSchemaPair

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueWildSchemaPair'):
            return visitor.visitFugueWildSchemaPair(self)
        else:
            return visitor.visitChildren(self)