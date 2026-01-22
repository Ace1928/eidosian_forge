from antlr4.atn.Transition import Transition
class DecisionState(ATNState):
    __slots__ = ('decision', 'nonGreedy')

    def __init__(self):
        super().__init__()
        self.decision = -1
        self.nonGreedy = False