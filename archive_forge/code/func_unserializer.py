import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
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