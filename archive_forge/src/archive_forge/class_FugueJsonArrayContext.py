from antlr4 import *
from io import StringIO
import sys
class FugueJsonArrayContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueJsonValue(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonValueContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonArray

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonArray'):
            return visitor.visitFugueJsonArray(self)
        else:
            return visitor.visitChildren(self)