from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def add_to_epsilon_closure(state_set, state):
    """
    Recursively add to |state_set| states reachable from the given state
    by epsilon moves.
    """
    if not state_set.get(state, 0):
        state_set[state] = 1
        state_set_2 = state.transitions.get_epsilon()
        if state_set_2:
            for state2 in state_set_2:
                add_to_epsilon_closure(state_set, state2)