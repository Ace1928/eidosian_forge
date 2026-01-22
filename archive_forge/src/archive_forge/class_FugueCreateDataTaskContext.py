from antlr4 import *
from io import StringIO
import sys
class FugueCreateDataTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.data = None
        self.schema = None

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def SCHEMA(self):
        return self.getToken(fugue_sqlParser.SCHEMA, 0)

    def fugueJsonArray(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonArrayContext, 0)

    def fugueSchema(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

    def DATA(self):
        return self.getToken(fugue_sqlParser.DATA, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueCreateDataTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueCreateDataTask'):
            return visitor.visitFugueCreateDataTask(self)
        else:
            return visitor.visitChildren(self)