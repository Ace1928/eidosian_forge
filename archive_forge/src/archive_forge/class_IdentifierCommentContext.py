from antlr4 import *
from io import StringIO
import sys
class IdentifierCommentContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def commentSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_identifierComment

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIdentifierComment'):
            return visitor.visitIdentifierComment(self)
        else:
            return visitor.visitChildren(self)