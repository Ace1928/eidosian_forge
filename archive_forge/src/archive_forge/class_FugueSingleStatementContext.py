from antlr4 import *
from io import StringIO
import sys
class FugueSingleStatementContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueSingleTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleTaskContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSingleStatement

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSingleStatement'):
            return visitor.visitFugueSingleStatement(self)
        else:
            return visitor.visitChildren(self)