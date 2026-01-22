from antlr4 import *
from io import StringIO
import sys
class IdentifierCommentListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifierComment(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierCommentContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierCommentContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_identifierCommentList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIdentifierCommentList'):
            return visitor.visitIdentifierCommentList(self)
        else:
            return visitor.visitChildren(self)