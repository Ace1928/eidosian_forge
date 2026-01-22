from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
def _source(self, *args):
    func = self.dispatch(*map(type, args))
    if not func:
        raise TypeError('No function found')
    return source(func)