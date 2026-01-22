import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _push_new_state(self):
    """Push a new state into history.

        This new state will be used to hold resolution results of the next
        coming round.
        """
    base = self._states[-1]
    state = State(mapping=base.mapping.copy(), criteria=base.criteria.copy(), backtrack_causes=base.backtrack_causes[:])
    self._states.append(state)