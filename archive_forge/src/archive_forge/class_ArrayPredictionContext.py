from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
class ArrayPredictionContext(PredictionContext):

    def __init__(self, parents: list, returnStates: list):
        super().__init__(calculateListsHashCode(parents, returnStates))
        self.parents = parents
        self.returnStates = returnStates

    def isEmpty(self):
        return self.returnStates[0] == PredictionContext.EMPTY_RETURN_STATE

    def __len__(self):
        return len(self.returnStates)

    def getParent(self, index: int):
        return self.parents[index]

    def getReturnState(self, index: int):
        return self.returnStates[index]

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, ArrayPredictionContext):
            return False
        elif hash(self) != hash(other):
            return False
        else:
            return self.returnStates == other.returnStates and self.parents == other.parents

    def __str__(self):
        if self.isEmpty():
            return '[]'
        with StringIO() as buf:
            buf.write('[')
            for i in range(0, len(self.returnStates)):
                if i > 0:
                    buf.write(', ')
                if self.returnStates[i] == PredictionContext.EMPTY_RETURN_STATE:
                    buf.write('$')
                    continue
                buf.write(str(self.returnStates[i]))
                if self.parents[i] is not None:
                    buf.write(' ')
                    buf.write(str(self.parents[i]))
                else:
                    buf.write('null')
            buf.write(']')
            return buf.getvalue()

    def __hash__(self):
        return self.cachedHashCode