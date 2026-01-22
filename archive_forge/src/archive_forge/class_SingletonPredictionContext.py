from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
class SingletonPredictionContext(PredictionContext):

    @staticmethod
    def create(parent: PredictionContext, returnState: int):
        if returnState == PredictionContext.EMPTY_RETURN_STATE and parent is None:
            return SingletonPredictionContext.EMPTY
        else:
            return SingletonPredictionContext(parent, returnState)

    def __init__(self, parent: PredictionContext, returnState: int):
        hashCode = calculateHashCode(parent, returnState)
        super().__init__(hashCode)
        self.parentCtx = parent
        self.returnState = returnState

    def __len__(self):
        return 1

    def getParent(self, index: int):
        return self.parentCtx

    def getReturnState(self, index: int):
        return self.returnState

    def __eq__(self, other):
        if self is other:
            return True
        elif other is None:
            return False
        elif not isinstance(other, SingletonPredictionContext):
            return False
        else:
            return self.returnState == other.returnState and self.parentCtx == other.parentCtx

    def __hash__(self):
        return self.cachedHashCode

    def __str__(self):
        up = '' if self.parentCtx is None else str(self.parentCtx)
        if len(up) == 0:
            if self.returnState == self.EMPTY_RETURN_STATE:
                return '$'
            else:
                return str(self.returnState)
        else:
            return str(self.returnState) + ' ' + up