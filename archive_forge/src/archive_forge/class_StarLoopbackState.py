from antlr4.atn.Transition import Transition
class StarLoopbackState(ATNState):

    def __init__(self):
        super().__init__()
        self.stateType = self.STAR_LOOP_BACK