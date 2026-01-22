from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _get_state_path(self, state, prefix=None):
    prefix = prefix or []
    if state in self.states.values():
        return prefix + [state.name]
    for name in self.states:
        with self(name):
            res = self._get_state_path(state, prefix=prefix + [name])
            if res:
                return res
    return []