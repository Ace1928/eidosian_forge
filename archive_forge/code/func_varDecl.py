from antlr4 import *
from io import StringIO
import sys
def varDecl(self):
    return self.getTypedRuleContext(AutolevParser.VarDeclContext, 0)