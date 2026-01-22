from antlr4 import *
from io import StringIO
import sys
class ColTypeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.colName = None

    def dataType(self):
        return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def commentSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_colType

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitColType'):
            return visitor.visitColType(self)
        else:
            return visitor.visitChildren(self)