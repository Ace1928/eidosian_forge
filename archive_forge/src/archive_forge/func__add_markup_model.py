from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _add_markup_model(self, markup):
    initial = markup.get('state', None)
    if markup['class-name'] == 'self':
        self.add_model(self, initial)
    else:
        mod_name, cls_name = markup['class-name'].rsplit('.', 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        self.add_model(cls(), initial)