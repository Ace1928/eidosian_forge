from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _remap_state(self, state, remap):
    drop_event = []
    remapped_transitions = []
    for evt in self.events.values():
        self.events[evt.name] = copy.copy(evt)
    for trigger, event in self.events.items():
        drop_source = []
        event.transitions = copy.deepcopy(event.transitions)
        for source_name, trans_source in event.transitions.items():
            if source_name in remap:
                drop_source.append(source_name)
                continue
            drop_trans = []
            for trans in trans_source:
                if trans.dest in remap:
                    conditions, unless = ([], [])
                    for cond in trans.conditions:
                        (unless, conditions)[cond.target].append(cond.func)
                    remapped_transitions.append({'trigger': trigger, 'source': state.name + self.state_cls.separator + trans.source, 'dest': remap[trans.dest], 'conditions': conditions, 'unless': unless, 'prepare': trans.prepare, 'before': trans.before, 'after': trans.after})
                    drop_trans.append(trans)
            for d_trans in drop_trans:
                trans_source.remove(d_trans)
            if not trans_source:
                drop_source.append(source_name)
        for d_source in drop_source:
            del event.transitions[d_source]
        if not event.transitions:
            drop_event.append(trigger)
    for d_event in drop_event:
        del self.events[d_event]
    return remapped_transitions