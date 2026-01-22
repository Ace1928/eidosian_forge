from antlr4 import *
from io import StringIO
import sys
class CtesContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def WITH(self):
        return self.getToken(fugue_sqlParser.WITH, 0)

    def namedQuery(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NamedQueryContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NamedQueryContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_ctes

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCtes'):
            return visitor.visitCtes(self)
        else:
            return visitor.visitChildren(self)