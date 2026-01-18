from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def verifyATN(self, atn: ATN):
    if not self.deserializationOptions.verifyATN:
        return
    for state in atn.states:
        if state is None:
            continue
        self.checkCondition(state.epsilonOnlyTransitions or len(state.transitions) <= 1)
        if isinstance(state, PlusBlockStartState):
            self.checkCondition(state.loopBackState is not None)
        if isinstance(state, StarLoopEntryState):
            self.checkCondition(state.loopBackState is not None)
            self.checkCondition(len(state.transitions) == 2)
            if isinstance(state.transitions[0].target, StarBlockStartState):
                self.checkCondition(isinstance(state.transitions[1].target, LoopEndState))
                self.checkCondition(not state.nonGreedy)
            elif isinstance(state.transitions[0].target, LoopEndState):
                self.checkCondition(isinstance(state.transitions[1].target, StarBlockStartState))
                self.checkCondition(state.nonGreedy)
            else:
                raise Exception('IllegalState')
        if isinstance(state, StarLoopbackState):
            self.checkCondition(len(state.transitions) == 1)
            self.checkCondition(isinstance(state.transitions[0].target, StarLoopEntryState))
        if isinstance(state, LoopEndState):
            self.checkCondition(state.loopBackState is not None)
        if isinstance(state, RuleStartState):
            self.checkCondition(state.stopState is not None)
        if isinstance(state, BlockStartState):
            self.checkCondition(state.endState is not None)
        if isinstance(state, BlockEndState):
            self.checkCondition(state.startState is not None)
        if isinstance(state, DecisionState):
            self.checkCondition(len(state.transitions) <= 1 or state.decision >= 0)
        else:
            self.checkCondition(len(state.transitions) <= 1 or isinstance(state, RuleStopState))