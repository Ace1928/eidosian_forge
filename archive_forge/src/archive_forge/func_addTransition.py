from antlr4.atn.Transition import Transition
def addTransition(self, trans: Transition, index: int=-1):
    if len(self.transitions) == 0:
        self.epsilonOnlyTransitions = trans.isEpsilon
    elif self.epsilonOnlyTransitions != trans.isEpsilon:
        self.epsilonOnlyTransitions = False
    if index == -1:
        self.transitions.append(trans)
    else:
        self.transitions.insert(index, trans)