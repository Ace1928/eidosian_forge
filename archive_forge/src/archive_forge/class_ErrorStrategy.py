import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
class ErrorStrategy(object):

    def reset(self, recognizer: Parser):
        pass

    def recoverInline(self, recognizer: Parser):
        pass

    def recover(self, recognizer: Parser, e: RecognitionException):
        pass

    def sync(self, recognizer: Parser):
        pass

    def inErrorRecoveryMode(self, recognizer: Parser):
        pass

    def reportError(self, recognizer: Parser, e: RecognitionException):
        pass