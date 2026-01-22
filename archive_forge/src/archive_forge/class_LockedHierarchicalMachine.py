from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
class LockedHierarchicalMachine(LockedMachine, HierarchicalMachine):
    """
        A threadsafe hierarchical machine.
    """
    event_cls = NestedEvent

    def _get_qualified_state_name(self, state):
        return self.get_global_name(state.name)