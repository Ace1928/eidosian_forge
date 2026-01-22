from antlr4 import *
from io import StringIO
import sys
class TheNamespaceContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def NAMESPACE(self):
        return self.getToken(fugue_sqlParser.NAMESPACE, 0)

    def DATABASE(self):
        return self.getToken(fugue_sqlParser.DATABASE, 0)

    def SCHEMA(self):
        return self.getToken(fugue_sqlParser.SCHEMA, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_theNamespace

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTheNamespace'):
            return visitor.visitTheNamespace(self)
        else:
            return visitor.visitChildren(self)