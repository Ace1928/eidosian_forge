import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _explicitize_args(func):
    if hasattr(func, 'func_code'):
        varnames = func.func_code.co_varnames
    else:
        varnames = func.__code__.co_varnames

    def wrapper(*args, **kwargs):
        if '_explicit_args' in kwargs:
            raise Exception('Variable _explicit_args should not be set.')
        kwargs['_explicit_args'] = list(set(list(varnames[:len(args)]) + [k for k, _ in kwargs.items()]))
        if 'self' in kwargs['_explicit_args']:
            kwargs['_explicit_args'].remove('self')
        return func(*args, **kwargs)
    if hasattr(inspect, 'signature'):
        new_sig = inspect.signature(wrapper).replace(parameters=inspect.signature(func).parameters.values())
        wrapper.__signature__ = new_sig
    return wrapper