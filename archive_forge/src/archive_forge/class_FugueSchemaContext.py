from antlr4 import *
from io import StringIO
import sys
class FugueSchemaContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueSchemaPair(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaPairContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaPairContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSchema

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchema'):
            return visitor.visitFugueSchema(self)
        else:
            return visitor.visitChildren(self)