from antlr4 import *
from io import StringIO
import sys
class ErrorCapturingUnitToUnitIntervalContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.body = None
        self.error1 = None
        self.error2 = None

    def unitToUnitInterval(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.UnitToUnitIntervalContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.UnitToUnitIntervalContext, i)

    def multiUnitsInterval(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultiUnitsIntervalContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_errorCapturingUnitToUnitInterval

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitErrorCapturingUnitToUnitInterval'):
            return visitor.visitErrorCapturingUnitToUnitInterval(self)
        else:
            return visitor.visitChildren(self)