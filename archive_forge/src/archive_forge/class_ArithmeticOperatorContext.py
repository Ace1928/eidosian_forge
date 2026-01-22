from antlr4 import *
from io import StringIO
import sys
class ArithmeticOperatorContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def SLASH(self):
        return self.getToken(fugue_sqlParser.SLASH, 0)

    def PERCENT(self):
        return self.getToken(fugue_sqlParser.PERCENT, 0)

    def DIV(self):
        return self.getToken(fugue_sqlParser.DIV, 0)

    def TILDE(self):
        return self.getToken(fugue_sqlParser.TILDE, 0)

    def AMPERSAND(self):
        return self.getToken(fugue_sqlParser.AMPERSAND, 0)

    def PIPE(self):
        return self.getToken(fugue_sqlParser.PIPE, 0)

    def CONCAT_PIPE(self):
        return self.getToken(fugue_sqlParser.CONCAT_PIPE, 0)

    def HAT(self):
        return self.getToken(fugue_sqlParser.HAT, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_arithmeticOperator

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitArithmeticOperator'):
            return visitor.visitArithmeticOperator(self)
        else:
            return visitor.visitChildren(self)