import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def _num_required_args(func):
    if func not in signatures:
        return None
    sigs = signatures[func]
    vals = [check_required_args(sig) for sig in sigs]
    val = vals[0]
    if all((x == val for x in vals)):
        return val
    return None