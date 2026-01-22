from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
def _get_qualified_state_name(self, state):
    return self.get_global_name(state.name)