from antlr4 import *
from io import StringIO
import sys
class MultipartIdentifierListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_multipartIdentifierList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMultipartIdentifierList'):
            return visitor.visitMultipartIdentifierList(self)
        else:
            return visitor.visitChildren(self)