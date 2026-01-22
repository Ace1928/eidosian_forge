from antlr4 import *
from io import StringIO
import sys
class FugueAssignmentContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.varname = None
        self.sign = None

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def fugueAssignmentSign(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentSignContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueAssignment

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueAssignment'):
            return visitor.visitFugueAssignment(self)
        else:
            return visitor.visitChildren(self)