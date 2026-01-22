from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _trigger_event_nested(self, event_data, trigger, _state_tree):
    model = event_data.model
    if _state_tree is None:
        _state_tree = self.build_state_tree(listify(getattr(model, self.model_attribute)), self.state_cls.separator)
    res = {}
    for key, value in _state_tree.items():
        if value:
            with self(key):
                tmp = self._trigger_event_nested(event_data, trigger, value)
                if tmp is not None:
                    res[key] = tmp
        if res.get(key, False) is False and trigger in self.events:
            event_data.event = self.events[trigger]
            tmp = event_data.event.trigger_nested(event_data)
            if tmp is not None:
                res[key] = tmp
    return None if not res or all((v is None for v in res.values())) else any(res.values())