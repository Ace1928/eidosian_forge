from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
class NestedState(State):
    """ A state which allows substates.
    Attributes:
        states (OrderedDict): A list of substates of the current state.
        events (dict): A list of events defined for the nested state.
        initial (list, str, NestedState or Enum): (Name of a) child or list of children that should be entered when the
        state is entered.
    """
    separator = '_'
    u" Separator between the names of parent and child states. In case '_' is required for\n        naming state, this value can be set to other values such as '.' or even unicode characters\n        such as '↦' (limited to Python 3 though).\n    "

    def __init__(self, name, on_enter=None, on_exit=None, ignore_invalid_triggers=None, initial=None):
        super(NestedState, self).__init__(name=name, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers)
        self.initial = initial
        self.events = {}
        self.states = OrderedDict()
        self._scope = []

    def add_substate(self, state):
        """ Adds a state as a substate.
        Args:
            state (NestedState): State to add to the current state.
        """
        self.add_substates(state)

    def add_substates(self, states):
        """ Adds a list of states to the current state.
        Args:
            states (list): List of state to add to the current state.
        """
        for state in listify(states):
            self.states[state.name] = state

    def scoped_enter(self, event_data, scope=None):
        """ Enters a state with the provided scope.
        Args:
            event_data (NestedEventData): The currently processed event.
            scope (list(str)): Names of the state's parents starting with the top most parent.
        """
        self._scope = scope or []
        try:
            self.enter(event_data)
        finally:
            self._scope = []

    def scoped_exit(self, event_data, scope=None):
        """ Exits a state with the provided scope.
        Args:
            event_data (NestedEventData): The currently processed event.
            scope (list(str)): Names of the state's parents starting with the top most parent.
        """
        self._scope = scope or []
        try:
            self.exit(event_data)
        finally:
            self._scope = []

    @property
    def name(self):
        return self.separator.join(self._scope + [super(NestedState, self).name])