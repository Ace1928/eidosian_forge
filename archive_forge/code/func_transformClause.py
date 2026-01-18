from antlr4 import *
from io import StringIO
import sys
def transformClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.TransformClauseContext, 0)