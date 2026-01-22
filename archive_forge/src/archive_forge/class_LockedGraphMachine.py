from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
class LockedGraphMachine(GraphMachine, LockedMachine):
    """
        A threadsafe machine with graph support.
    """

    @staticmethod
    def format_references(func):
        if isinstance(func, partial) and func.func.__name__.startswith('_locked_method'):
            func = func.args[0]
        return GraphMachine.format_references(func)