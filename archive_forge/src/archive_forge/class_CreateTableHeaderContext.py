from antlr4 import *
from io import StringIO
import sys
class CreateTableHeaderContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def TEMPORARY(self):
        return self.getToken(fugue_sqlParser.TEMPORARY, 0)

    def EXTERNAL(self):
        return self.getToken(fugue_sqlParser.EXTERNAL, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_createTableHeader

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateTableHeader'):
            return visitor.visitCreateTableHeader(self)
        else:
            return visitor.visitChildren(self)