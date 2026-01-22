from antlr4 import *
from io import StringIO
import sys
class AddTableColumnsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.columns = None
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def ADD(self):
        return self.getToken(fugue_sqlParser.ADD, 0)

    def COLUMN(self):
        return self.getToken(fugue_sqlParser.COLUMN, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def qualifiedColTypeWithPositionList(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAddTableColumns'):
            return visitor.visitAddTableColumns(self)
        else:
            return visitor.visitChildren(self)