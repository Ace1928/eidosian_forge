from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class PredicateTransition(AbstractPredicateTransition):
    __slots__ = ('serializationType', 'ruleIndex', 'predIndex', 'isCtxDependent')

    def __init__(self, target: ATNState, ruleIndex: int, predIndex: int, isCtxDependent: bool):
        super().__init__(target)
        self.serializationType = self.PREDICATE
        self.ruleIndex = ruleIndex
        self.predIndex = predIndex
        self.isCtxDependent = isCtxDependent
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return False

    def getPredicate(self):
        return Predicate(self.ruleIndex, self.predIndex, self.isCtxDependent)

    def __str__(self):
        return 'pred_' + str(self.ruleIndex) + ':' + str(self.predIndex)