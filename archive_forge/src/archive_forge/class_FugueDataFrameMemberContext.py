from antlr4 import *
from io import StringIO
import sys
class FugueDataFrameMemberContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.index = None
        self.key = None

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueDataFrameMember

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFrameMember'):
            return visitor.visitFugueDataFrameMember(self)
        else:
            return visitor.visitChildren(self)