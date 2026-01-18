from antlr4 import *
from io import StringIO
import sys
def windowClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)