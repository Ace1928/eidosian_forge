from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def addState(self, state: ATNState):
    if state is not None:
        state.atn = self
        state.stateNumber = len(self.states)
    self.states.append(state)