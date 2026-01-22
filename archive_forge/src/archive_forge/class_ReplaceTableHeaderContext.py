from antlr4 import *
from io import StringIO
import sys
class ReplaceTableHeaderContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def OR(self):
        return self.getToken(fugue_sqlParser.OR, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_replaceTableHeader

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitReplaceTableHeader'):
            return visitor.visitReplaceTableHeader(self)
        else:
            return visitor.visitChildren(self)