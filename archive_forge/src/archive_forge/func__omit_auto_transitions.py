from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _omit_auto_transitions(self, event):
    return self.auto_transitions_markup is False and self._is_auto_transition(event)