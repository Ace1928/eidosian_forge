from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def add_substate(self, state):
    """ Adds a state as a substate.
        Args:
            state (NestedState): State to add to the current state.
        """
    self.add_substates(state)