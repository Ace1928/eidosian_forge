from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class RuleTransition(Transition):
    __slots__ = ('ruleIndex', 'precedence', 'followState', 'serializationType')

    def __init__(self, ruleStart: RuleStartState, ruleIndex: int, precedence: int, followState: ATNState):
        super().__init__(ruleStart)
        self.ruleIndex = ruleIndex
        self.precedence = precedence
        self.followState = followState
        self.serializationType = self.RULE
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return False