from antlr4 import *
from io import StringIO
import sys
class DescribeFuncNameContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def comparisonOperator(self):
        return self.getTypedRuleContext(fugue_sqlParser.ComparisonOperatorContext, 0)

    def arithmeticOperator(self):
        return self.getTypedRuleContext(fugue_sqlParser.ArithmeticOperatorContext, 0)

    def predicateOperator(self):
        return self.getTypedRuleContext(fugue_sqlParser.PredicateOperatorContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_describeFuncName

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDescribeFuncName'):
            return visitor.visitDescribeFuncName(self)
        else:
            return visitor.visitChildren(self)