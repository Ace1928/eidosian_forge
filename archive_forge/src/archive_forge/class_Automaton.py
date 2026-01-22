from itertools import chain
class Automaton(object):
    """
    A declaration of a finite state machine.

    Note that this is not the machine itself; it is immutable.
    """

    def __init__(self):
        """
        Initialize the set of transitions and the initial state.
        """
        self._initialState = _NO_STATE
        self._transitions = set()

    @property
    def initialState(self):
        """
        Return this automaton's initial state.
        """
        return self._initialState

    @initialState.setter
    def initialState(self, state):
        """
        Set this automaton's initial state.  Raises a ValueError if
        this automaton already has an initial state.
        """
        if self._initialState is not _NO_STATE:
            raise ValueError('initial state already set to {}'.format(self._initialState))
        self._initialState = state

    def addTransition(self, inState, inputSymbol, outState, outputSymbols):
        """
        Add the given transition to the outputSymbol. Raise ValueError if
        there is already a transition with the same inState and inputSymbol.
        """
        for anInState, anInputSymbol, anOutState, _ in self._transitions:
            if anInState == inState and anInputSymbol == inputSymbol:
                raise ValueError('already have transition from {} via {}'.format(inState, inputSymbol))
        self._transitions.add((inState, inputSymbol, outState, tuple(outputSymbols)))

    def allTransitions(self):
        """
        All transitions.
        """
        return frozenset(self._transitions)

    def inputAlphabet(self):
        """
        The full set of symbols acceptable to this automaton.
        """
        return {inputSymbol for inState, inputSymbol, outState, outputSymbol in self._transitions}

    def outputAlphabet(self):
        """
        The full set of symbols which can be produced by this automaton.
        """
        return set(chain.from_iterable((outputSymbols for inState, inputSymbol, outState, outputSymbols in self._transitions)))

    def states(self):
        """
        All valid states; "Q" in the mathematical description of a state
        machine.
        """
        return frozenset(chain.from_iterable(((inState, outState) for inState, inputSymbol, outState, outputSymbol in self._transitions)))

    def outputForInput(self, inState, inputSymbol):
        """
        A 2-tuple of (outState, outputSymbols) for inputSymbol.
        """
        for anInState, anInputSymbol, outState, outputSymbols in self._transitions:
            if (inState, inputSymbol) == (anInState, anInputSymbol):
                return (outState, list(outputSymbols))
        raise NoTransition(state=inState, symbol=inputSymbol)