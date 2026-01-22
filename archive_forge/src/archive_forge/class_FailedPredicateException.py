from antlr3.constants import INVALID_TOKEN_TYPE
class FailedPredicateException(RecognitionException):
    """@brief A semantic predicate failed during validation.

    Validation of predicates
    occurs when normally parsing the alternative just like matching a token.
    Disambiguating predicate evaluation occurs when we hoist a predicate into
    a prediction decision.
    """

    def __init__(self, input, ruleName, predicateText):
        RecognitionException.__init__(self, input)
        self.ruleName = ruleName
        self.predicateText = predicateText

    def __str__(self):
        return 'FailedPredicateException(' + self.ruleName + ',{' + self.predicateText + '}?)'
    __repr__ = __str__