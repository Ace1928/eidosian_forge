from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Recognizer import Recognizer
class InputMismatchException(RecognitionException):

    def __init__(self, recognizer: Parser):
        super().__init__(recognizer=recognizer, input=recognizer.getInputStream(), ctx=recognizer._ctx)
        self.offendingToken = recognizer.getCurrentToken()