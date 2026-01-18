from antlr4 import *
from io import StringIO
import sys
def whenClause(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.WhenClauseContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.WhenClauseContext, i)