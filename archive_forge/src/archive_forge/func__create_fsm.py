import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
@staticmethod
def _create_fsm(start_state, add_start=True, hierarchical=False, add_states=None):
    if hierarchical:
        m = machines.HierarchicalFiniteMachine()
    else:
        m = machines.FiniteMachine()
    if add_start:
        m.add_state(start_state)
        m.default_start_state = start_state
    if add_states:
        for s in add_states:
            if s not in m:
                m.add_state(s)
    return m