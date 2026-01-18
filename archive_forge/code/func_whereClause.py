from antlr4 import *
from io import StringIO
import sys
def whereClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)