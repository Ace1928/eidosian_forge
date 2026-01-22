from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _add_machine_states(self, state, remap):
    new_states = [s for s in state.states.values() if remap is None or s not in remap]
    self.add_states(new_states)
    for evt in state.events.values():
        self.events[evt.name] = evt
    if self.scoped.initial is None:
        self.scoped.initial = state.initial