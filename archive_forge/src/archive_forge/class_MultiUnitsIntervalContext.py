from antlr4 import *
from io import StringIO
import sys
class MultiUnitsIntervalContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def intervalValue(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IntervalValueContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IntervalValueContext, i)

    def intervalUnit(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IntervalUnitContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IntervalUnitContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_multiUnitsInterval

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMultiUnitsInterval'):
            return visitor.visitMultiUnitsInterval(self)
        else:
            return visitor.visitChildren(self)