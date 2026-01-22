from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
class EmptyPredictionContext(SingletonPredictionContext):

    def __init__(self):
        super().__init__(None, PredictionContext.EMPTY_RETURN_STATE)

    def isEmpty(self):
        return True

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return self.cachedHashCode

    def __str__(self):
        return '$'