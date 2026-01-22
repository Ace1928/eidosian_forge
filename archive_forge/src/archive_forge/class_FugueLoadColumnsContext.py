from antlr4 import *
from io import StringIO
import sys
class FugueLoadColumnsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.schema = None
        self.cols = None

    def fugueSchema(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueLoadColumns

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueLoadColumns'):
            return visitor.visitFugueLoadColumns(self)
        else:
            return visitor.visitChildren(self)