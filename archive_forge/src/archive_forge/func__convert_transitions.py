from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _convert_transitions(self, root):
    root['transitions'] = []
    for event in self.events.values():
        if self._omit_auto_transitions(event):
            continue
        for transitions in event.transitions.items():
            for trans in transitions[1]:
                t_def = _convert(trans, self.transition_attributes, self.format_references)
                t_def['trigger'] = event.name
                con = [x for x in (rep(f.func, self.format_references) for f in trans.conditions if f.target) if x]
                unl = [x for x in (rep(f.func, self.format_references) for f in trans.conditions if not f.target) if x]
                if con:
                    t_def['conditions'] = con
                if unl:
                    t_def['unless'] = unl
                root['transitions'].append(t_def)