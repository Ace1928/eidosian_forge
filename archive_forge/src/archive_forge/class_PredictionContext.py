from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
class PredictionContext(object):
    EMPTY = None
    EMPTY_RETURN_STATE = 2147483647
    globalNodeCount = 1
    id = globalNodeCount

    def __init__(self, cachedHashCode: int):
        self.cachedHashCode = cachedHashCode

    def __len__(self):
        return 0

    def isEmpty(self):
        return self is self.EMPTY

    def hasEmptyPath(self):
        return self.getReturnState(len(self) - 1) == self.EMPTY_RETURN_STATE

    def getReturnState(self, index: int):
        raise IllegalStateException('illegal!')

    def __hash__(self):
        return self.cachedHashCode