import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def add_states(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
    """ Calls the base method and regenerates all models's graphs. """
    super(GraphMachine, self).add_states(states, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
    for model in self.models:
        model.get_graph(force_new=True)