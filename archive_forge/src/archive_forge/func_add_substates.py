from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def add_substates(self, states):
    """ Adds a list of states to the current state.
        Args:
            states (list): List of state to add to the current state.
        """
    for state in listify(states):
        self.states[state.name] = state