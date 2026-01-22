from antlr4 import *
from io import StringIO
import sys
class SkewSpecContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def SKEWED(self):
        return self.getToken(fugue_sqlParser.SKEWED, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def identifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def constantList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ConstantListContext, 0)

    def nestedConstantList(self):
        return self.getTypedRuleContext(fugue_sqlParser.NestedConstantListContext, 0)

    def STORED(self):
        return self.getToken(fugue_sqlParser.STORED, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def DIRECTORIES(self):
        return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_skewSpec

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSkewSpec'):
            return visitor.visitSkewSpec(self)
        else:
            return visitor.visitChildren(self)