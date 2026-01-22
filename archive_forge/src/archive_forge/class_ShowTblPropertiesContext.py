from antlr4 import *
from io import StringIO
import sys
class ShowTblPropertiesContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.table = None
        self.key = None
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def TBLPROPERTIES(self):
        return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def tablePropertyKey(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyKeyContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowTblProperties'):
            return visitor.visitShowTblProperties(self)
        else:
            return visitor.visitChildren(self)