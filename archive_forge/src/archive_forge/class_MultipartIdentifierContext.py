from antlr4 import *
from io import StringIO
import sys
class MultipartIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self._errorCapturingIdentifier = None
        self.parts = list()

    def errorCapturingIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_multipartIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMultipartIdentifier'):
            return visitor.visitMultipartIdentifier(self)
        else:
            return visitor.visitChildren(self)