from antlr4 import *
from io import StringIO
import sys
class FunctionIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.db = None
        self.function = None

    def errorCapturingIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_functionIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFunctionIdentifier'):
            return visitor.visitFunctionIdentifier(self)
        else:
            return visitor.visitChildren(self)