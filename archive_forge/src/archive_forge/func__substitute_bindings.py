import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _substitute_bindings(fstruct, bindings, fs_class, visited):
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for fname, fval in items:
        while isinstance(fval, Variable) and fval in bindings:
            fval = fstruct[fname] = bindings[fval]
        if isinstance(fval, fs_class):
            _substitute_bindings(fval, bindings, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            fstruct[fname] = fval.substitute_bindings(bindings)