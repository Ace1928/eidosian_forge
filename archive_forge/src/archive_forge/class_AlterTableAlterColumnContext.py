from antlr4 import *
from io import StringIO
import sys
class AlterTableAlterColumnContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.table = None
        self.column = None
        self.copyFrom(ctx)

    def ALTER(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.ALTER)
        else:
            return self.getToken(fugue_sqlParser.ALTER, i)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def CHANGE(self):
        return self.getToken(fugue_sqlParser.CHANGE, 0)

    def COLUMN(self):
        return self.getToken(fugue_sqlParser.COLUMN, 0)

    def alterColumnAction(self):
        return self.getTypedRuleContext(fugue_sqlParser.AlterColumnActionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAlterTableAlterColumn'):
            return visitor.visitAlterTableAlterColumn(self)
        else:
            return visitor.visitChildren(self)