from antlr4 import *
from io import StringIO
import sys
class FugueSingleOutputExtensionCommonContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.fugueUsing = None
        self.params = None
        self.schema = None

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def fugueExtension(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

    def SCHEMA(self):
        return self.getToken(fugue_sqlParser.SCHEMA, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def fugueSchema(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSingleOutputExtensionCommon

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSingleOutputExtensionCommon'):
            return visitor.visitFugueSingleOutputExtensionCommon(self)
        else:
            return visitor.visitChildren(self)