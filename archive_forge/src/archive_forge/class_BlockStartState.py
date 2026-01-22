from antlr4.atn.Transition import Transition
class BlockStartState(DecisionState):
    __slots__ = 'endState'

    def __init__(self):
        super().__init__()
        self.endState = None