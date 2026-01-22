from antlr4 import *
from io import StringIO
import sys
class WindowFrameContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.frameType = None
        self.start = None
        self.end = None

    def RANGE(self):
        return self.getToken(fugue_sqlParser.RANGE, 0)

    def frameBound(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FrameBoundContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FrameBoundContext, i)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def BETWEEN(self):
        return self.getToken(fugue_sqlParser.BETWEEN, 0)

    def AND(self):
        return self.getToken(fugue_sqlParser.AND, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_windowFrame

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWindowFrame'):
            return visitor.visitWindowFrame(self)
        else:
            return visitor.visitChildren(self)