import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def update_object_dict(v):
    changed = False
    rv = dict(v.__dict__)
    for key in rv.keys():
        if key not in v._nonvar_fields:
            prior = rv[key]
            rv[key] = cls.apply(fn, prior, cache, skip_fn)
            changed = changed or prior is not rv[key]
    if changed:
        return v.clone(**rv)
    return v