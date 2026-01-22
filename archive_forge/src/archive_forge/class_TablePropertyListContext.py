from antlr4 import *
from io import StringIO
import sys
class TablePropertyListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def tableProperty(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tablePropertyList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTablePropertyList'):
            return visitor.visitTablePropertyList(self)
        else:
            return visitor.visitChildren(self)