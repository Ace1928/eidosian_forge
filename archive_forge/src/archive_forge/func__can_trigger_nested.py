from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _can_trigger_nested(self, model, trigger, path, *args, **kwargs):
    evt = NestedEventData(None, None, self, model, args, kwargs)
    if trigger in self.events:
        source_path = copy.copy(path)
        while source_path:
            state_name = self.state_cls.separator.join(source_path)
            for transition in self.events[trigger].transitions.get(state_name, []):
                try:
                    _ = self.get_state(transition.dest)
                except ValueError:
                    continue
                self.callbacks(self.prepare_event, evt)
                self.callbacks(transition.prepare, evt)
                if all((c.check(evt) for c in transition.conditions)):
                    return True
            source_path.pop(-1)
    if path:
        with self(path.pop(0)):
            return self._can_trigger_nested(model, trigger, path, *args, **kwargs)
    return False