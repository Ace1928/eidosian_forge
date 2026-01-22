from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _identify_callback(self, name):
    callback_type, target = super(MarkupMachine, self)._identify_callback(name)
    if callback_type:
        self._needs_update = True
    return (callback_type, target)