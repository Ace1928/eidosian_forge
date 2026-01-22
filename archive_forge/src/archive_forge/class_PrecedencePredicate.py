from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO
class PrecedencePredicate(SemanticContext):

    def __init__(self, precedence: int=0):
        self.precedence = precedence

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        return parser.precpred(outerContext, self.precedence)

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        if parser.precpred(outerContext, self.precedence):
            return SemanticContext.NONE
        else:
            return None

    def __lt__(self, other):
        return self.precedence < other.precedence

    def __hash__(self):
        return 31

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, PrecedencePredicate):
            return False
        else:
            return self.precedence == other.precedence