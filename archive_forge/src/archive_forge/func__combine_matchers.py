import collections
import copy
import itertools
import random
import re
import warnings
def _combine_matchers(target, kwargs, require_spec):
    """Merge target specifications with keyword arguments (PRIVATE).

    Dispatch the components to the various matcher functions, then merge into a
    single boolean function.
    """
    if not target:
        if not kwargs:
            if require_spec:
                raise ValueError('you must specify a target object or keyword arguments.')
            return lambda x: True
        return _attribute_matcher(kwargs)
    match_obj = _object_matcher(target)
    if not kwargs:
        return match_obj
    match_kwargs = _attribute_matcher(kwargs)
    return lambda x: match_obj(x) and match_kwargs(x)