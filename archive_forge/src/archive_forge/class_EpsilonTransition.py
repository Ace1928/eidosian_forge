from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class EpsilonTransition(Transition):
    __slots__ = ('serializationType', 'outermostPrecedenceReturn')

    def __init__(self, target, outermostPrecedenceReturn=-1):
        super(EpsilonTransition, self).__init__(target)
        self.serializationType = self.EPSILON
        self.isEpsilon = True
        self.outermostPrecedenceReturn = outermostPrecedenceReturn

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return False

    def __str__(self):
        return 'epsilon'