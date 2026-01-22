from antlr4.atn.Transition import Transition
class BasicBlockStartState(BlockStartState):

    def __init__(self):
        super().__init__()
        self.stateType = self.BLOCK_START