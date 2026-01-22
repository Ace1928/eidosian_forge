from antlr4 import *
from io import StringIO
import sys
class FugueSingleFileContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.single = None

    def SINGLE(self):
        return self.getToken(fugue_sqlParser.SINGLE, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSingleFile

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSingleFile'):
            return visitor.visitFugueSingleFile(self)
        else:
            return visitor.visitChildren(self)