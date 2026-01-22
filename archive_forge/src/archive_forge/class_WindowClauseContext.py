from antlr4 import *
from io import StringIO
import sys
class WindowClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def WINDOW(self):
        return self.getToken(fugue_sqlParser.WINDOW, 0)

    def namedWindow(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NamedWindowContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NamedWindowContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_windowClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWindowClause'):
            return visitor.visitWindowClause(self)
        else:
            return visitor.visitChildren(self)