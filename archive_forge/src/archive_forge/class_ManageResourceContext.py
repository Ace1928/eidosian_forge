from antlr4 import *
from io import StringIO
import sys
class ManageResourceContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.op = None
        self.copyFrom(ctx)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def ADD(self):
        return self.getToken(fugue_sqlParser.ADD, 0)

    def LIST(self):
        return self.getToken(fugue_sqlParser.LIST, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitManageResource'):
            return visitor.visitManageResource(self)
        else:
            return visitor.visitChildren(self)