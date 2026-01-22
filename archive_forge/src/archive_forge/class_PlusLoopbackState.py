from antlr4.atn.Transition import Transition
class PlusLoopbackState(DecisionState):

    def __init__(self):
        super().__init__()
        self.stateType = self.PLUS_LOOP_BACK