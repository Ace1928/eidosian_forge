from antlr4 import *
from io import StringIO
import sys
class PartitionValContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def EQUAL(self):
        return self.getToken(fugue_sqlParser.EQUAL, 0)

    def constant(self):
        return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_partitionVal

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPartitionVal'):
            return visitor.visitPartitionVal(self)
        else:
            return visitor.visitChildren(self)