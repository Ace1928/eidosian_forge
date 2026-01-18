from antlr4 import *
from io import StringIO
import sys
def unitToUnitInterval(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.UnitToUnitIntervalContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.UnitToUnitIntervalContext, i)