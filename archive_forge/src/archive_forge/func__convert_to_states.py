import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
def _convert_to_states(state_space):
    for state in state_space:
        if isinstance(state, dict):
            state = State(**state)
        yield state