from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
class LockedHierarchicalGraphMachine(GraphMachine, LockedHierarchicalMachine):
    """
        A threadsafe hierarchical machine with graph support.
    """
    transition_cls = NestedGraphTransition
    event_cls = NestedEvent

    @staticmethod
    def format_references(func):
        if isinstance(func, partial) and func.func.__name__.startswith('_locked_method'):
            func = func.args[0]
        return GraphMachine.format_references(func)