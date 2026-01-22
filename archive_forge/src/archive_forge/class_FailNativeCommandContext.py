from antlr4 import *
from io import StringIO
import sys
class FailNativeCommandContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def ROLE(self):
        return self.getToken(fugue_sqlParser.ROLE, 0)

    def unsupportedHiveNativeCommands(self):
        return self.getTypedRuleContext(fugue_sqlParser.UnsupportedHiveNativeCommandsContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFailNativeCommand'):
            return visitor.visitFailNativeCommand(self)
        else:
            return visitor.visitChildren(self)