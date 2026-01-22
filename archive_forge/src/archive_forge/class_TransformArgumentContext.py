from antlr4 import *
from io import StringIO
import sys
class TransformArgumentContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def constant(self):
        return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_transformArgument

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTransformArgument'):
            return visitor.visitTransformArgument(self)
        else:
            return visitor.visitChildren(self)