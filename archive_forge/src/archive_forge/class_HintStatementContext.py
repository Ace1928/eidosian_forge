from antlr4 import *
from io import StringIO
import sys
class HintStatementContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.hintName = None
        self._primaryExpression = None
        self.parameters = list()

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def primaryExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PrimaryExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_hintStatement

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitHintStatement'):
            return visitor.visitHintStatement(self)
        else:
            return visitor.visitChildren(self)