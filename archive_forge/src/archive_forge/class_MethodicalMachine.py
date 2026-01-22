import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
class MethodicalMachine(object):
    """
    A :class:`MethodicalMachine` is an interface to an `Automaton`
    that uses methods on a class.
    """

    def __init__(self):
        self._automaton = Automaton()
        self._reducers = {}
        self._symbol = gensym()

    def __get__(self, oself, type=None):
        """
        L{MethodicalMachine} is an implementation detail for setting up
        class-level state; applications should never need to access it on an
        instance.
        """
        if oself is not None:
            raise AttributeError('MethodicalMachine is an implementation detail.')
        return self

    @_keywords_only
    def state(self, initial=False, terminal=False, serialized=None):
        """
        Declare a state, possibly an initial state or a terminal state.

        This is a decorator for methods, but it will modify the method so as
        not to be callable any more.

        :param bool initial: is this state the initial state?
            Only one state on this :class:`automat.MethodicalMachine`
            may be an initial state; more than one is an error.

        :param bool terminal: Is this state a terminal state?
            i.e. a state that the machine can end up in?
            (This is purely informational at this point.)

        :param Hashable serialized: a serializable value
            to be used to represent this state to external systems.
            This value should be hashable;
            :py:func:`unicode` is a good type to use.
        """

        def decorator(stateMethod):
            state = MethodicalState(machine=self, method=stateMethod, serialized=serialized)
            if initial:
                self._automaton.initialState = state
            return state
        return decorator

    @_keywords_only
    def input(self):
        """
        Declare an input.

        This is a decorator for methods.
        """

        def decorator(inputMethod):
            return MethodicalInput(automaton=self._automaton, method=inputMethod, symbol=self._symbol)
        return decorator

    @_keywords_only
    def output(self):
        """
        Declare an output.

        This is a decorator for methods.

        This method will be called when the state machine transitions to this
        state as specified in the decorated `output` method.
        """

        def decorator(outputMethod):
            return MethodicalOutput(machine=self, method=outputMethod)
        return decorator

    def _oneTransition(self, startState, inputToken, endState, outputTokens, collector):
        """
        See L{MethodicalState.upon}.
        """
        self._automaton.addTransition(startState, inputToken, endState, tuple(outputTokens))
        inputToken.collectors[startState] = collector

    @_keywords_only
    def serializer(self):
        """

        """

        def decorator(decoratee):

            @wraps(decoratee)
            def serialize(oself):
                transitioner = _transitionerFromInstance(oself, self._symbol, self._automaton)
                return decoratee(oself, transitioner._state.serialized)
            return serialize
        return decorator

    @_keywords_only
    def unserializer(self):
        """

        """

        def decorator(decoratee):

            @wraps(decoratee)
            def unserialize(oself, *args, **kwargs):
                state = decoratee(oself, *args, **kwargs)
                mapping = {}
                for eachState in self._automaton.states():
                    mapping[eachState.serialized] = eachState
                transitioner = _transitionerFromInstance(oself, self._symbol, self._automaton)
                transitioner._state = mapping[state]
                return None
            return unserialize
        return decorator

    @property
    def _setTrace(self):
        return MethodicalTracer(self._automaton, self._symbol)

    def asDigraph(self):
        """
        Generate a L{graphviz.Digraph} that represents this machine's
        states and transitions.

        @return: L{graphviz.Digraph} object; for more information, please
            see the documentation for
            U{graphviz<https://graphviz.readthedocs.io/>}

        """
        from ._visualize import makeDigraph
        return makeDigraph(self._automaton, stateAsString=lambda state: state.method.__name__, inputAsString=lambda input: input.method.__name__, outputAsString=lambda output: output.method.__name__)