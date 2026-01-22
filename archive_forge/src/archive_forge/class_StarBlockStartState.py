from antlr4.atn.Transition import Transition
class StarBlockStartState(BlockStartState):

    def __init__(self):
        super().__init__()
        self.stateType = self.STAR_BLOCK_START