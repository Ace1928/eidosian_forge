from antlr4 import *
from io import StringIO
import sys
class MultiInsertQueryBodyContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def insertInto(self):
        return self.getTypedRuleContext(fugue_sqlParser.InsertIntoContext, 0)

    def fromStatementBody(self):
        return self.getTypedRuleContext(fugue_sqlParser.FromStatementBodyContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_multiInsertQueryBody

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMultiInsertQueryBody'):
            return visitor.visitMultiInsertQueryBody(self)
        else:
            return visitor.visitChildren(self)