from antlr4 import *
from io import StringIO
import sys
def ID_action(self, localctx: RuleContext, actionIndex: int):
    if actionIndex == 0:
        char = self.text[0]
        if char.isupper():
            self.type = XPathLexer.TOKEN_REF
        else:
            self.type = XPathLexer.RULE_REF