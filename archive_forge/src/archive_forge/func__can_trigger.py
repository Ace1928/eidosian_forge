from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _can_trigger(self, model, trigger, *args, **kwargs):
    state_tree = self.build_state_tree(getattr(model, self.model_attribute), self.state_cls.separator)
    ordered_states = resolve_order(state_tree)
    for state_path in ordered_states:
        with self():
            return self._can_trigger_nested(model, trigger, state_path, *args, **kwargs)