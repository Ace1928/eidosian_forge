from antlr4 import *
from io import StringIO
import sys
class FugueSaveModeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def TO(self):
        return self.getToken(fugue_sqlParser.TO, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def APPEND(self):
        return self.getToken(fugue_sqlParser.APPEND, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSaveMode

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSaveMode'):
            return visitor.visitFugueSaveMode(self)
        else:
            return visitor.visitChildren(self)